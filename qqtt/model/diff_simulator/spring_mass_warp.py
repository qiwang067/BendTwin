import torch
from qqtt.utils import logger, cfg
import warp as wp

wp.init()
wp.set_device("cuda:0")
if not cfg.use_graph:
    wp.config.mode = "debug"
    wp.config.verbose = True
    wp.config.verify_autograd_array_access = True


class State:
    '''
    State class to hold the state of the spring-mass system at each substep.
    It is subjected to further modifications as now we want to include extra 
    mechanical properties like torque, bending, etc.
    '''
    def __init__(self, wp_init_vertices, num_control_points):
        self.wp_x = wp.zeros_like(wp_init_vertices, requires_grad=True)
        self.wp_v_before_collision = wp.zeros_like(
            wp_init_vertices, 
            requires_grad=True
            )    # velocity before collision
        self.wp_v_before_ground = wp.zeros_like(
            wp_init_vertices, 
            requires_grad=True
            )    # velocity before ground collision
        self.wp_v = wp.zeros_like(
            self.wp_x, 
            requires_grad=True
            )    # velocity after collision
        self.wp_vertice_forces = wp.zeros_like(
            self.wp_x, 
            requires_grad=True
            )    # forces on each point
        # No need to compute the gradient for the control points
        self.wp_control_x = wp.zeros(
            (num_control_points), dtype=wp.vec3, requires_grad=False
        )
        self.wp_control_v = wp.zeros_like(self.wp_control_x, requires_grad=False)

    def clear_forces(self):
        self.wp_vertice_forces.zero_()

    # This takes more time but not necessary, will be overwritten directly
    # def clear_control(self):
    #     self.wp_control_x.zero_()
    #     self.wp_control_v.zero_()

    # def clear_states(self):
    #     self.wp_x.zero_()
    #     self.wp_v_before_ground.zero_()
    #     self.wp_v.zero_()

    @property
    def requires_grad(self):
        """Indicates whether the state arrays have gradient computation enabled."""
        return self.wp_x.requires_grad


ANGLE_EPS = 1e-8
INVALID_SPRING_VALUE = -1.0
INT32_MAX = 2**31 - 1


def compute_initial_spring_angles(
    init_vertices, init_springs, num_object_points, controller_points
):
    """
    Compute initial angle information at both endpoints of each spring in parallel with Warp.

    Args:
        init_vertices: [N_total, 3] initial vertex positions, including object and control points.
        init_springs: [M, 2] spring connectivity [point_i, point_j].
        num_object_points: Number of object mass points, excluding control points.
        controller_points: Control-point coordinates; may be None.

    Returns:
        spring_angle_info: [M, 4] angle information for each spring.
                          [:, 0]: Angle in radians from A->B to A's reference direction.
                          [:, 1]: Angle in radians from B->A to B's reference direction.
                          [:, 2]: Index of A's reference neighbor.
                          [:, 3]: Index of B's reference neighbor.
    """
    import torch

    device = init_vertices.device
    n_vertices = init_vertices.shape[0]
    n_springs = init_springs.shape[0]

    spring_angle_info = torch.full(
        (n_springs, 4), INVALID_SPRING_VALUE, dtype=torch.float32, device=device
    )
    if n_vertices == 0 or n_springs == 0:
        return spring_angle_info

    # Warp expects int32 indices for vec2i, convert if necessary
    if init_springs.dtype != torch.int32:
        springs_tensor = init_springs.to(torch.int32)
    else:
        springs_tensor = init_springs

    vertices_tensor = init_vertices.contiguous()
    if springs_tensor.device != vertices_tensor.device:
        springs_tensor = springs_tensor.to(vertices_tensor.device)
    springs_tensor = springs_tensor.contiguous()
    spring_angle_info = spring_angle_info.contiguous()

    wp_vertices = wp.from_torch(vertices_tensor, dtype=wp.vec3, requires_grad=False)
    wp_springs = wp.from_torch(springs_tensor, dtype=wp.vec2i, requires_grad=False)
    wp_spring_angle_info = wp.from_torch(
        spring_angle_info, dtype=wp.float32, requires_grad=False
    )

    wp_device = wp_vertices.device
    wp_ref_neighbors = wp.full(
        (n_vertices,),
        INT32_MAX,
        dtype=wp.int32,
        requires_grad=False,
        device=wp_device,
    )
    wp_min_spring = wp.full(
        (n_vertices,),
        INT32_MAX,
        dtype=wp.int32,
        requires_grad=False,
        device=wp_device,
    )

    wp.launch(
        build_vertex_reference_info,
        dim=n_springs,
        inputs=[wp_springs, n_vertices, wp_ref_neighbors, wp_min_spring],
        device=wp_device,
    )
    wp.launch(
        finalize_vertex_metadata,
        dim=n_vertices,
        inputs=[wp_ref_neighbors, wp_min_spring],
        device=wp_device,
    )
    wp.launch(
        compute_spring_angle_kernel,
        dim=n_springs,
        inputs=[
            wp_vertices,
            wp_springs,
            wp_ref_neighbors,
            wp_min_spring,
            wp_spring_angle_info,
            n_vertices,
        ],
        device=wp_device,
    )
    wp.synchronize_device()

    return spring_angle_info


@wp.func
def safe_normalize(vec: wp.vec3):
    length = wp.length(vec)
    if length > ANGLE_EPS:
        return vec / length
    return wp.vec3(0.0, 0.0, 0.0)


@wp.func
def compute_angle(center: wp.vec3, neighbor: wp.vec3, reference: wp.vec3):
    dir_ref = safe_normalize(reference - center)
    dir_neighbor = safe_normalize(neighbor - center)
    cos_theta = wp.clamp(wp.dot(dir_ref, dir_neighbor), -1.0, 1.0)
    return wp.acos(cos_theta)


@wp.kernel(enable_backward=False)
def build_vertex_reference_info(
    springs: wp.array(dtype=wp.vec2i),
    num_vertices: int,
    ref_neighbors: wp.array(dtype=wp.int32),
    min_spring_per_vertex: wp.array(dtype=wp.int32),
):
    tid = wp.tid()
    p1 = springs[tid][0]
    p2 = springs[tid][1]

    if p1 < 0 or p2 < 0:
        return
    if p1 >= num_vertices or p2 >= num_vertices:
        return

    wp.atomic_min(min_spring_per_vertex, p1, tid)
    wp.atomic_min(min_spring_per_vertex, p2, tid)
    wp.atomic_min(ref_neighbors, p1, p2)
    wp.atomic_min(ref_neighbors, p2, p1)


@wp.kernel(enable_backward=False)
def finalize_vertex_metadata(
    ref_neighbors: wp.array(dtype=wp.int32),
    min_spring_per_vertex: wp.array(dtype=wp.int32),
):
    tid = wp.tid()

    if ref_neighbors[tid] == INT32_MAX:
        ref_neighbors[tid] = -1
    if min_spring_per_vertex[tid] == INT32_MAX:
        min_spring_per_vertex[tid] = -1


@wp.kernel(enable_backward=False)
def compute_spring_angle_kernel(
    vertices: wp.array(dtype=wp.vec3),
    springs: wp.array(dtype=wp.vec2i),
    ref_neighbors: wp.array(dtype=wp.int32),
    min_spring_per_vertex: wp.array(dtype=wp.int32),
    spring_angle_info: wp.array2d(dtype=float),
    num_vertices: int,
):
    tid = wp.tid()
    p1 = springs[tid][0]
    p2 = springs[tid][1]

    if p1 < 0 or p2 < 0:
        return
    if p1 >= num_vertices or p2 >= num_vertices:
        return

    ref_p1 = ref_neighbors[p1]
    ref_p2 = ref_neighbors[p2]

    pos_p1 = vertices[p1]
    pos_p2 = vertices[p2]

    if ref_p1 >= 0 and ref_p1 < num_vertices and min_spring_per_vertex[p1] == tid:
        pos_ref_p1 = vertices[ref_p1]
        angle_p1 = compute_angle(pos_p1, pos_p2, pos_ref_p1)
        spring_angle_info[tid, 0] = angle_p1
        spring_angle_info[tid, 2] = wp.float32(ref_p1)
    else:
        spring_angle_info[tid, 0] = INVALID_SPRING_VALUE
        spring_angle_info[tid, 2] = INVALID_SPRING_VALUE

    if ref_p2 >= 0 and ref_p2 < num_vertices and min_spring_per_vertex[p2] == tid:
        pos_ref_p2 = vertices[ref_p2]
        angle_p2 = compute_angle(pos_p2, pos_p1, pos_ref_p2)
        spring_angle_info[tid, 1] = angle_p2
        spring_angle_info[tid, 3] = wp.float32(ref_p2)
    else:
        spring_angle_info[tid, 1] = INVALID_SPRING_VALUE
        spring_angle_info[tid, 3] = INVALID_SPRING_VALUE


@wp.kernel(enable_backward=False)
def copy_vec3(data: wp.array(dtype=wp.vec3), origin: wp.array(dtype=wp.vec3)):
    tid = wp.tid()
    origin[tid] = data[tid]


@wp.kernel(enable_backward=False)
def copy_int(data: wp.array(dtype=wp.int32), origin: wp.array(dtype=wp.int32)):
    tid = wp.tid()
    origin[tid] = data[tid]


@wp.kernel(enable_backward=False)
def copy_float(data: wp.array(dtype=wp.float32), origin: wp.array(dtype=wp.float32)):
    tid = wp.tid()
    origin[tid] = data[tid]


@wp.kernel(enable_backward=False)
def set_control_points(
    num_substeps: int,
    original_control_point: wp.array(dtype=wp.vec3),
    target_control_point: wp.array(dtype=wp.vec3),
    step: int,
    control_x: wp.array(dtype=wp.vec3),
):
    # Set the control points in each substep
    tid = wp.tid()

    t = float(step + 1) / float(num_substeps)
    control_x[tid] = (
        original_control_point[tid]
        + (target_control_point[tid] - original_control_point[tid]) * t
    )


@wp.kernel
def eval_springs(
    x: wp.array(dtype=wp.vec3),              # Current positions of object points
    v: wp.array(dtype=wp.vec3),              # Current velocities of object points
    control_x: wp.array(dtype=wp.vec3),      # Current controller positions (interpolated)
    control_v: wp.array(dtype=wp.vec3),      # Current controller velocities
    num_object_points: int,                  # Number of object points (vs. total vertices)
    springs: wp.array(dtype=wp.vec2i),       # Spring connectivity [spring_id] -> [point1, point2]
    rest_lengths: wp.array(dtype=float),     # Natural length of each spring
    spring_Y: wp.array(dtype=float),         # Stiffness per spring (in log space)
    dashpot_damping: float,                  # Velocity-based damping coefficient
    spring_Y_min: float,                     # Minimum stiffness limit
    spring_Y_max: float,                     # Maximum stiffness limit
    f: wp.array(dtype=wp.vec3),             # Output: force accumulator per object point
):
    # 1. Thread Assignment & Spring Validation
    tid = wp.tid()

    # 2. Spring Force Computation
    if wp.exp(spring_Y[tid]) > spring_Y_min:
        idx1 = springs[tid][0]  # First vertex of spring
        idx2 = springs[tid][1]  # Second vertex of spring

        # Handle vertex types (object vs. controller)
        if idx1 >= num_object_points:
            # means that it is a controller point
            x1 = control_x[idx1 - num_object_points]  # Controller position
            v1 = control_v[idx1 - num_object_points]  # Controller velocity
        else:
            x1 = x[idx1]  # Object position
            v1 = v[idx1]  # Object velocity

        if idx2 >= num_object_points:
            x2 = control_x[idx2 - num_object_points]  # Controller position
            v2 = control_v[idx2 - num_object_points]  # Controller velocity
        else:
            x2 = x[idx2]  # Object position  
            v2 = v[idx2]  # Object velocity

        # calculate the spring force using Hooke's law
        rest = rest_lengths[tid]

        dis = x2 - x1
        dis_len = wp.length(dis)

        d = dis / wp.max(dis_len, 1e-6)  # Unit direction vector

        spring_force = (
            wp.clamp(wp.exp(spring_Y[tid]), low=spring_Y_min, high=spring_Y_max)
            * (dis_len / rest - 1.0)
            * d
        )    # we need to add extra torque term for bending here
        
        # damping force based on relative velocity 
        # the energy dissipated by dashpot damping 
        # prevents oscillation
        v_rel = wp.dot(v2 - v1, d)    # Relative velocity along spring direction
        dashpot_forces = dashpot_damping * v_rel * d

        overall_force = spring_force + dashpot_forces

        if idx1 < num_object_points:
            wp.atomic_add(f, idx1, overall_force)
        if idx2 < num_object_points:
            wp.atomic_sub(f, idx2, overall_force)


@wp.kernel
def eval_bending_forces(
    x: wp.array(dtype=wp.vec3),                  # Current positions of object points
    v: wp.array(dtype=wp.vec3),                  # Current velocities of object points
    control_x: wp.array(dtype=wp.vec3),          # Current controller positions
    control_v: wp.array(dtype=wp.vec3),          # Current controller velocities
    num_object_points: int,                      # Number of object points
    springs: wp.array(dtype=wp.vec2i),           # Spring connectivity [spring_id] -> [point1, point2]
    spring_angle_info: wp.array2d(dtype=float),  # Spring angle info [n_springs, 4]
    bending_stiffness: wp.array(dtype=float),    # Bending stiffness per vertex (in log space)
    bend_damping: float,                         # Bending damping coefficient
    bend_stiffness_min: float,                   # Minimum bending stiffness limit
    bend_stiffness_max: float,                   # Maximum bending stiffness limit
    f: wp.array(dtype=wp.vec3),                  # Output: force accumulator per object point
):
    """
    Compute bending forces in parallel per spring at both endpoint particles.

    Data layout:
    - spring_angle_info[spring_id, 0]: Initial angle from A->B to A's reference direction (A is the center, B the neighbor).
    - spring_angle_info[spring_id, 1]: Initial angle from B->A to B's reference direction (B is the center, A the neighbor).
    - spring_angle_info[spring_id, 2]: Index of A's reference neighbor.
    - spring_angle_info[spring_id, 3]: Index of B's reference neighbor.
    - bend_damping: Bending damping coefficient; larger values produce stronger damping.

    Velocity data (v, control_v) is used to compute tangential angular velocity and apply velocity-dependent bending damping.
    """
    spring_id = wp.tid()  # Current spring index

    # Get particle indices at both spring endpoints (idx_1, idx_2).
    p1 = springs[spring_id][0]
    p2 = springs[spring_id][1]

    # Get angle information for this spring.
    rest_angle_p1 = spring_angle_info[spring_id, 0]  # Initial A->B angle
    rest_angle_p2 = spring_angle_info[spring_id, 1]  # Initial B->A angle
    ref_p1 = int(spring_angle_info[spring_id, 2])    # A's reference neighbor
    ref_p2 = int(spring_angle_info[spring_id, 3])    # B's reference neighbor

    invalid_p1 = rest_angle_p1 < 0.0 or ref_p1 < 0
    invalid_p2 = rest_angle_p2 < 0.0 or ref_p2 < 0
    if invalid_p1 and invalid_p2:
        return

    process_p1 = (not invalid_p1) and p1 < num_object_points and ref_p1 != p2
    process_p2 = (not invalid_p2) and p2 < num_object_points and ref_p2 != p1

    # === Process the bending force at endpoint p1 ===
    # The current frame uses p1 as the center, p2 as the neighbor, and ref_p1 as the reference neighbor.
    # rest_angle_p1 is the ori_p2--p1--ori_ref_p1 angle.
    # ori_p2 is p2's initial position, and ori_ref_p1 is ref_p1's initial position.
    # The current angle is p2--p1--ref_p1.
    if process_p1:
        # Convert from log space and clamp to the configured bounds.
        k_bend_p1 = wp.clamp(
            wp.exp(bending_stiffness[p1]), 
            low=bend_stiffness_min, 
            high=bend_stiffness_max
            )

        # Get the position of p1.
        pos_p1 = x[p1]
        vel_p1 = v[p1]

        # Get the position of p1's reference neighbor.
        if ref_p1 < num_object_points:
            pos_ref_p1 = x[ref_p1]
            vel_ref_p1 = v[ref_p1]
        else:
            idx_ref_p1 = ref_p1 - num_object_points
            pos_ref_p1 = control_x[idx_ref_p1]
            vel_ref_p1 = control_v[idx_ref_p1]

        # Get the position of p2.
        if p2 < num_object_points:
            pos_p2 = x[p2]
            vel_p2 = v[p2]
        else:
            idx_p2 = p2 - num_object_points
            pos_p2 = control_x[idx_p2]
            vel_p2 = control_v[idx_p2]

        # Compute the reference direction vector.
        ref_vec_p1 = pos_ref_p1 - pos_p1
        ref_len_p1 = wp.length(ref_vec_p1)

        # Compute the current spring direction vector.
        cur_vec_p1 = pos_p2 - pos_p1
        cur_len_p1 = wp.length(cur_vec_p1)

        ref_vec_p1_norm = ref_vec_p1 / ref_len_p1  # u
        cur_vec_p1_norm = cur_vec_p1 / cur_len_p1  # v

        # Compute the current angle (cos(\theta) = u·v); u and v are unit vectors.
        cos_angle_p1 = wp.clamp(
            wp.dot(ref_vec_p1_norm, cur_vec_p1_norm), 
            -1.0, 1.0
            ) # cos theta
        current_angle_p1 = wp.acos(cos_angle_p1)

        # Compute the angular deviation (\theta - \theta_0).
        angle_diff_p1 = current_angle_p1 - rest_angle_p1

        if wp.abs(angle_diff_p1) > 1e-6:
            # Compute the force direction.
            cross_axis_p1 = wp.cross(ref_vec_p1_norm, cur_vec_p1_norm)    # sin<u,v>
            cross_axis_p1_norm = cross_axis_p1 / wp.length(cross_axis_p1)
            force_dir_p1 = wp.cross(cross_axis_p1_norm, cur_vec_p1_norm)
            force_dir_ref_p1 = wp.cross(ref_vec_p1_norm, cross_axis_p1_norm)
            force_dir_ref_p1 = force_dir_ref_p1 / wp.length(force_dir_ref_p1)

            # Compute the elastic force on the current rod induced by k_bend.
            elastic_force_mag_cur_p1 = k_bend_p1 * angle_diff_p1 / cur_len_p1
            elastic_force_vec_cur_p1 = elastic_force_mag_cur_p1 * force_dir_p1
            wp.atomic_add(f, p1, elastic_force_vec_cur_p1)
            if p2 < num_object_points:
                wp.atomic_add(f, p2, -elastic_force_vec_cur_p1)

            if ref_p1 < num_object_points:            
                # Compute the elastic force on the reference rod induced by k_bend.
                elastic_force_mag_ref_p1 = k_bend_p1 * angle_diff_p1 / ref_len_p1
                elastic_force_vec_ref_p1 = elastic_force_mag_ref_p1 * force_dir_ref_p1
                wp.atomic_add(f, p1, -elastic_force_vec_ref_p1)
                wp.atomic_add(f, ref_p1, elastic_force_vec_ref_p1)

            # 2. Compute relative angular velocity as tangential velocity divided by rod length.
            rel_vel_cur_p1 = vel_p2 - vel_p1
            omega_cur_p1 = wp.dot(rel_vel_cur_p1, force_dir_p1) / cur_len_p1

            rel_vel_ref_p1 = vel_ref_p1 - vel_p1
            omega_ref_p1 = wp.dot(rel_vel_ref_p1, force_dir_ref_p1) / ref_len_p1
            angular_vel_p1 = omega_cur_p1 - omega_ref_p1

            # 3A. Compute the damping force on the current rod.
            target_damping_p1 = bend_damping * angular_vel_p1
            damping_force_mag_cur_p1 = wp.clamp(
                target_damping_p1,
                -wp.abs(elastic_force_mag_cur_p1),
                wp.abs(elastic_force_mag_cur_p1),
            )
            damping_force_vec_cur_p1 = damping_force_mag_cur_p1 * force_dir_p1
            wp.atomic_add(f, p1, -damping_force_vec_cur_p1)
            if p2 < num_object_points:
                wp.atomic_add(f, p2, damping_force_vec_cur_p1)

            if ref_p1 < num_object_points:
                # 3B. Compute the damping force on the reference rod.
                damping_force_mag_ref_p1 = wp.clamp(
                    target_damping_p1,
                    -wp.abs(elastic_force_mag_ref_p1),
                    wp.abs(elastic_force_mag_ref_p1),
                )
                damping_force_vec_ref_p1 = damping_force_mag_ref_p1 * force_dir_ref_p1
                wp.atomic_add(f, p1, -damping_force_vec_ref_p1)
                wp.atomic_add(f, ref_p1, damping_force_vec_ref_p1)

    # === Process the bending force at endpoint p2 ===
    if process_p2:  
        # Convert from log space and clamp to the configured bounds.
        k_bend_p2 = wp.clamp(wp.exp(bending_stiffness[p2]), low=bend_stiffness_min, high=bend_stiffness_max)

        # Get the position of p2.
        pos_p2_2 = x[p2]
        vel_p2 = v[p2]

        # Get the position of p2's reference neighbor.
        if ref_p2 < num_object_points:
            pos_ref_p2 = x[ref_p2]
            vel_ref_p2 = v[ref_p2]
        else:
            idx_ref_p2 = ref_p2 - num_object_points
            pos_ref_p2 = control_x[idx_ref_p2]
            vel_ref_p2 = control_v[idx_ref_p2]

        # Get the position of p1.
        if p1 < num_object_points:
            pos_p1_2 = x[p1]
            vel_p1 = v[p1]
        else:
            pos_p1_2 = control_x[p1 - num_object_points]
            vel_p1 = control_v[p1 - num_object_points]

        ref_vec_p2 = pos_ref_p2 - pos_p2_2
        ref_len_p2 = wp.length(ref_vec_p2)
        cur_vec_p2 = pos_p1_2 - pos_p2_2
        cur_len_p2 = wp.length(cur_vec_p2)

        ref_vec_p2_norm = ref_vec_p2 / ref_len_p2
        cur_vec_p2_norm = cur_vec_p2 / cur_len_p2

        cos_angle_p2 = wp.clamp(wp.dot(ref_vec_p2_norm, cur_vec_p2_norm), -1.0, 1.0)
        current_angle_p2 = wp.acos(cos_angle_p2)
        angle_diff_p2 = current_angle_p2 - rest_angle_p2

        if wp.abs(angle_diff_p2) > 1e-6:
            cross_axis_p2 = wp.cross(ref_vec_p2_norm, cur_vec_p2_norm)
            cross_axis_p2_norm = cross_axis_p2 / wp.length(cross_axis_p2)
            force_dir_p2 = wp.cross(cross_axis_p2_norm, cur_vec_p2_norm)
            force_dir_ref_p2 = wp.cross(ref_vec_p2_norm, cross_axis_p2_norm)
            force_dir_ref_p2 = force_dir_ref_p2 / wp.length(force_dir_ref_p2)

            # 1A. Compute the elastic force on the current rod induced by k_bend.
            elastic_force_mag_cur_p2 = k_bend_p2 * angle_diff_p2 / cur_len_p2
            # 1B. Compute the elastic force on the reference rod induced by k_bend.
            elastic_force_mag_ref_p2 = k_bend_p2 * angle_diff_p2 / ref_len_p2

            elastic_force_vec_cur_p2 = elastic_force_mag_cur_p2 * force_dir_p2
            wp.atomic_add(f, p2, elastic_force_vec_cur_p2)
            if p1 < num_object_points:
                wp.atomic_sub(f, p1, elastic_force_vec_cur_p2)

            if ref_p2 < num_object_points:
                elastic_force_vec_ref_p2 = elastic_force_mag_ref_p2 * force_dir_ref_p2
                wp.atomic_add(f, p2, -elastic_force_vec_ref_p2)
                wp.atomic_add(f, ref_p2, elastic_force_vec_ref_p2)

            rel_vel_cur_p2 = vel_p1 - vel_p2
            omega_cur_p2 = wp.dot(rel_vel_cur_p2, force_dir_p2) / cur_len_p2

            rel_vel_ref_p2 = vel_ref_p2 - vel_p2
            omega_ref_p2 = wp.dot(rel_vel_ref_p2, force_dir_ref_p2) / ref_len_p2
            angular_vel_p2 = omega_cur_p2 - omega_ref_p2

            # 3A. Compute the damping force on the current rod.
            target_damping_p2 = bend_damping * angular_vel_p2
            damping_force_mag_cur_p2 = wp.clamp(
                target_damping_p2,
                -wp.abs(elastic_force_mag_cur_p2),
                wp.abs(elastic_force_mag_cur_p2),
            )
            damping_force_vec_cur_p2 = damping_force_mag_cur_p2 * force_dir_p2
            wp.atomic_add(f, p2, -damping_force_vec_cur_p2)
            if p1 < num_object_points:
                wp.atomic_add(f, p1, damping_force_vec_cur_p2)

            if ref_p2 < num_object_points:
                # 3B. Compute the damping force on the reference rod.
                damping_force_mag_ref_p2 = wp.clamp(
                    target_damping_p2,
                    -wp.abs(elastic_force_mag_ref_p2),
                    wp.abs(elastic_force_mag_ref_p2),
                )
                damping_force_vec_ref_p2 = damping_force_mag_ref_p2 * force_dir_ref_p2
                wp.atomic_add(f, p2, -damping_force_vec_ref_p2)
                wp.atomic_add(f, ref_p2, damping_force_vec_ref_p2)


@wp.kernel
def update_vel_from_force(
    v: wp.array(dtype=wp.vec3),
    f: wp.array(dtype=wp.vec3),
    masses: wp.array(dtype=wp.float32),
    dt: float,
    drag_damping: float,
    reverse_factor: float,
    v_new: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()

    v0 = v[tid]    # current velocity
    f0 = f[tid]    # current force
    m0 = masses[tid]   # unit mass

    # drag damping factor in exponential decay
    drag_damping_factor = wp.exp(-dt * drag_damping)
    # spring force + damping force + gravity force
    all_force = f0 + m0 * wp.vec3(0.0, 0.0, -9.8) * reverse_factor
    # a = F/m
    a = all_force / m0
    v1 = v0 + a * dt               # semi-implicit Euler
    v2 = v1 * drag_damping_factor  # apply drag damping

    v_new[tid] = v2


@wp.func
def loop(
    i: int,                                          # Current vertex index(tid)
    collision_indices: wp.array2d(dtype=wp.int32),   # Precomputed neighbor indices [N, 500]
    collision_number: wp.array(dtype=wp.int32),      # Number of neighbors per vertex [N]
    x: wp.array(dtype=wp.vec3),                      # Current positions
    v: wp.array(dtype=wp.vec3),                      # Current velocities
    masses: wp.array(dtype=wp.float32),              # Vertex masses
    masks: wp.array(dtype=wp.int32),                 # Collision group IDs
    collision_dist: float,                           # Collision threshold
    clamp_collide_object_elas: float,                # Clamped elasticity [0,1]
    clamp_collide_object_fric: float,                # Clamped friction [0,2]
):
    x1 = x[i]
    v1 = v[i]
    m1 = masses[i]
    mask1 = masks[i]    # current point collision group ID (e.g, object A or B)

    valid_count = float(0.0)         # number of valid collisions
    J_sum = wp.vec3(0.0, 0.0, 0.0)

    for k in range(collision_number[i]):    # all neighbor points colliding with point i
        index = collision_indices[i][k]
        x2 = x[index]   # neighbor point position
        v2 = v[index]   # neighbor point velocity
        m2 = masses[index]
        mask2 = masks[index]

        dis = x2 - x1
        dis_len = wp.length(dis)
        relative_v = v2 - v1   # relative velocity between two points
        # If the distance is less than the collision distance and the two points are moving towards each other
        if (
            mask1 != mask2    # only consider inter-object collisions
            and dis_len < collision_dist
            # moving towards each other, otherwise no collision cause they are separating
            and wp.dot(dis, relative_v) < -1e-4    
        ):
            valid_count += 1.0
            
            # Unit direction vector representing collision normal (e.g., x2 - x1)
            collision_normal = dis / wp.max(dis_len, 1e-6) 

            # Decompose relative velocity into (1) normal and (2) tangential components
            # (1) normal component
            v_rel_n = wp.dot(relative_v, collision_normal) * collision_normal  
            impulse_n = (-(1.0 + clamp_collide_object_elas) * v_rel_n) / (
                1.0 / m1 + 1.0 / m2
            )  # simplified normal pounding models

            v_rel_n_length = wp.length(v_rel_n)  # magnitude of normal relative velocity

            # (2) tangential component
            v_rel_t = relative_v - v_rel_n       
            v_rel_t_length = wp.max(wp.length(v_rel_t), 1e-6)
            # To satisfy Coulomb friction model (F_friction ≤ μ * F_normal)
            # μ = friction * (1 + elasticity) * v_normal_magnitude / v_tangential_magnitude
            a = wp.max(
                0.0,
                1.0
                - clamp_collide_object_fric
                * (1.0 + clamp_collide_object_elas)
                * v_rel_n_length
                / v_rel_t_length,
            )
            
            impulse_t = (a - 1.0) * v_rel_t / (1.0 / m1 + 1.0 / m2)

            # Combine normal and tangential impulses
            J = impulse_n + impulse_t

            # Accumulate impulses from all collision neighbors
            J_sum += J

    return valid_count, J_sum


@wp.kernel(enable_backward=False)
def update_potential_collision(
    x: wp.array(dtype=wp.vec3),              # Current positions
    masks: wp.array(dtype=wp.int32),         # Collision group IDs
    collision_dist: float,                   # Collision threshold
    grid: wp.uint64,                         # Hash grid handle
    collision_indices: wp.array2d(dtype=wp.int32),  # Output: neighbor indices
    collision_number: wp.array(dtype=wp.int32),     # Output: neighbor counts
):
    '''
       It returns the index map for every point to 
       its potential collision points.
    '''
    tid = wp.tid()

    # order threads by cell
    # get the actual point index
    i = wp.hash_grid_point_id(grid, tid)

    x1 = x[i]         # current point position
    mask1 = masks[i]  # current point collision group ID

    neighbors = wp.hash_grid_query(grid, x1, collision_dist * 5.0) 
    for index in neighbors:
        if index != i:
            x2 = x[index]
            mask2 = masks[index]

            dis = x2 - x1
            dis_len = wp.length(dis)
            # If the distance is less than the collision distance and the two points are moving towards each other
            if mask1 != mask2 and dis_len < collision_dist:
                collision_indices[i][collision_number[i]] = index
                collision_number[i] += 1


@wp.kernel
def object_collision(
    x: wp.array(dtype=wp.vec3),                      # Current positions
    v: wp.array(dtype=wp.vec3),                      # Velocities before collision
    masses: wp.array(dtype=wp.float32),              # Point masses
    masks: wp.array(dtype=wp.int32),                 # Collision group IDs
    collide_object_elas: wp.array(dtype=float),      # Inter-object elasticity (CMA-ES param 7)
    collide_object_fric: wp.array(dtype=float),      # Inter-object friction (CMA-ES param 8)
    collision_dist: float,                           # Collision threshold (CMA-ES param 9)
    collision_indices: wp.array2d(dtype=wp.int32),   # Precomputed collision neighbors
    collision_number: wp.array(dtype=wp.int32),      # Number of neighbors per point
    v_new: wp.array(dtype=wp.vec3),                  # Output: velocities after collision
):
    tid = wp.tid()

    v1 = v[tid]           # velocity before collision
    m1 = masses[tid]      # unit mass

    clamp_collide_object_elas = wp.clamp(
        collide_object_elas[0], 
        low=0.0, 
        high=1.0 
        )                 # clamp the elasticity between 0 and 1(normal dir?)
    clamp_collide_object_fric = wp.clamp(
        collide_object_fric[0], 
        low=0.0, 
        high=2.0
        )                 # clamp the friction between 0 and 2 (tangential dir?)
    # J_sum: total impulse from all collisions 
    valid_count, J_sum = loop(
        tid,
        collision_indices,
        collision_number,
        x,
        v,
        masses,
        masks,
        collision_dist,
        clamp_collide_object_elas,
        clamp_collide_object_fric,
    )

    if valid_count > 0:
        # average impulse from all collisions
        J_average = J_sum / valid_count
        # F*t / m = delta_v
        v_new[tid] = v1 - J_average / m1
    else:
        v_new[tid] = v1


@wp.kernel
def integrate_ground_collision(
    x: wp.array(dtype=wp.vec3),              # Current positions
    v: wp.array(dtype=wp.vec3),              # Velocities (after object collisions)
    collide_elas: wp.array(dtype=float),     # Ground elasticity (CMA-ES param 5)
    collide_fric: wp.array(dtype=float),     # Ground friction (CMA-ES param 6)
    dt: float,                               # Timestep size
    reverse_factor: float,                   # Coordinate system orientation
    x_new: wp.array(dtype=wp.vec3),          # Output: new positions
    v_new: wp.array(dtype=wp.vec3),          # Output: new velocities
):
    tid = wp.tid()

    x0 = x[tid]
    v0 = v[tid]

    # Ground normal direction
    normal = wp.vec3(0.0, 0.0, 1.0) * reverse_factor 

    # Predict next z position
    x_z = x0[2]  
    v_z = v0[2]
    next_x_z = (x_z + v_z * dt) * reverse_factor

    if next_x_z < 0.0 and v_z * reverse_factor < -1e-4:
        # ground collision will happen within this timestep

        # Ground Collision
        # (1) Normal direction of velocity
        v_normal = wp.dot(v0, normal) * normal 
        v_normal_length = wp.length(v_normal)
        # (2) Tangential direction of velocity
        v_tao = v0 - v_normal                 
        v_tao_length = wp.max(wp.length(v_tao), 1e-6)
        # Clamp elasticity and friction coefficients
        clamp_collide_elas = wp.clamp(collide_elas[0], low=0.0, high=1.0)
        clamp_collide_fric = wp.clamp(collide_fric[0], low=0.0, high=2.0)

        # Update velocity based on collision response(Ground is static, infinite mass)
        v_normal_new = -clamp_collide_elas * v_normal 
        a = wp.max(
            0.0,
            1.0
            - clamp_collide_fric
            * (1.0 + clamp_collide_elas)
            * v_normal_length
            / v_tao_length,
        )
        v_tao_new = a * v_tao

        v1 = v_normal_new + v_tao_new
        # toi = -x_z / v_z  # Time when object reaches ground plane
        toi = -x_z / v_z
    else:
        v1 = v0
        toi = 0.0

    x_new[tid] = x0 + v0 * toi + v1 * (dt - toi)
    v_new[tid] = v1


@wp.kernel(enable_backward=False)
def compute_distances(
    pred: wp.array(dtype=wp.vec3),
    gt: wp.array(dtype=wp.vec3),
    gt_mask: wp.array(dtype=wp.int32),
    distances: wp.array2d(dtype=float),
):
    i, j = wp.tid()
    if gt_mask[i] == 1:
        dist = wp.length(gt[i] - pred[j])
        distances[i, j] = dist
    else:
        distances[i, j] = 1e6


@wp.kernel(enable_backward=False)
def compute_neigh_indices(
    distances: wp.array2d(dtype=float),
    neigh_indices: wp.array(dtype=wp.int32),
):
    i = wp.tid()
    min_dist = float(1e6)
    min_index = int(-1)
    for j in range(distances.shape[1]):
        if distances[i, j] < min_dist:
            min_dist = distances[i, j]
            min_index = j
    neigh_indices[i] = min_index


@wp.kernel
def compute_chamfer_loss(
    pred: wp.array(dtype=wp.vec3),
    gt: wp.array(dtype=wp.vec3),
    gt_mask: wp.array(dtype=wp.int32),
    num_valid: int,
    neigh_indices: wp.array(dtype=wp.int32),
    loss_weight: float,
    chamfer_loss: wp.array(dtype=float),
):
    i = wp.tid()
    if gt_mask[i] == 1:   # only consider the valid points
        min_pred = pred[neigh_indices[i]]      # the closest predicted point
        # L2 distance between the two nearest points in pred and gt
        min_dist = wp.length(min_pred - gt[i]) 
        final_min_dist = loss_weight * min_dist * min_dist / float(num_valid)
        wp.atomic_add(chamfer_loss, 0, final_min_dist)


@wp.kernel
def compute_track_loss(
    pred: wp.array(dtype=wp.vec3),
    gt: wp.array(dtype=wp.vec3),
    gt_mask: wp.array(dtype=wp.int32),
    num_valid: int,
    loss_weight: float,
    track_loss: wp.array(dtype=float),
):
    i = wp.tid()
    if gt_mask[i] == 1:
        # Calculate the smooth l1 loss modifed from fvcore.nn.smooth_l1_loss
        pred_x = pred[i][0]
        pred_y = pred[i][1]
        pred_z = pred[i][2]
        gt_x = gt[i][0]
        gt_y = gt[i][1]
        gt_z = gt[i][2]

        dist_x = wp.abs(pred_x - gt_x)
        dist_y = wp.abs(pred_y - gt_y)
        dist_z = wp.abs(pred_z - gt_z)

        if dist_x < 1.0:
            temp_track_loss_x = 0.5 * (dist_x**2.0)
        else:
            temp_track_loss_x = dist_x - 0.5

        if dist_y < 1.0:
            temp_track_loss_y = 0.5 * (dist_y**2.0)
        else:
            temp_track_loss_y = dist_y - 0.5

        if dist_z < 1.0:
            temp_track_loss_z = 0.5 * (dist_z**2.0)
        else:
            temp_track_loss_z = dist_z - 0.5

        temp_track_loss = temp_track_loss_x + temp_track_loss_y + temp_track_loss_z

        average_factor = float(num_valid) * 3.0

        final_track_loss = loss_weight * temp_track_loss / average_factor

        wp.atomic_add(track_loss, 0, final_track_loss)


@wp.kernel(enable_backward=False)
def set_int(input: int, output: wp.array(dtype=wp.int32)):
    output[0] = input


@wp.kernel(enable_backward=False)
def update_acc(
    v1: wp.array(dtype=wp.vec3),
    v2: wp.array(dtype=wp.vec3),
    prev_acc: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    prev_acc[tid] = v2[tid] - v1[tid]


@wp.kernel
def compute_acc_loss(
    v1: wp.array(dtype=wp.vec3),
    v2: wp.array(dtype=wp.vec3),
    prev_acc: wp.array(dtype=wp.vec3),
    num_object_points: int,
    acc_count: wp.array(dtype=wp.int32),
    acc_weight: float,
    acc_loss: wp.array(dtype=wp.float32),
):
    if acc_count[0] == 1:
        # Calculate the smooth l1 loss modifed from fvcore.nn.smooth_l1_loss
        tid = wp.tid()
        cur_acc = v2[tid] - v1[tid]
        cur_x = cur_acc[0]
        cur_y = cur_acc[1]
        cur_z = cur_acc[2]

        prev_x = prev_acc[tid][0]
        prev_y = prev_acc[tid][1]
        prev_z = prev_acc[tid][2]

        dist_x = wp.abs(cur_x - prev_x)
        dist_y = wp.abs(cur_y - prev_y)
        dist_z = wp.abs(cur_z - prev_z)

        if dist_x < 1.0:
            temp_acc_loss_x = 0.5 * (dist_x**2.0)
        else:
            temp_acc_loss_x = dist_x - 0.5

        if dist_y < 1.0:
            temp_acc_loss_y = 0.5 * (dist_y**2.0)
        else:
            temp_acc_loss_y = dist_y - 0.5

        if dist_z < 1.0:
            temp_acc_loss_z = 0.5 * (dist_z**2.0)
        else:
            temp_acc_loss_z = dist_z - 0.5

        temp_acc_loss = temp_acc_loss_x + temp_acc_loss_y + temp_acc_loss_z

        average_factor = float(num_object_points) * 3.0

        final_acc_loss = acc_weight * temp_acc_loss / average_factor

        wp.atomic_add(acc_loss, 0, final_acc_loss)


@wp.kernel
def compute_final_loss(
    chamfer_loss: wp.array(dtype=wp.float32),
    track_loss: wp.array(dtype=wp.float32),
    acc_loss: wp.array(dtype=wp.float32),
    loss: wp.array(dtype=wp.float32),
):
    loss[0] = chamfer_loss[0] + track_loss[0] + acc_loss[0]


@wp.kernel
def compute_simple_loss(
    pred: wp.array(dtype=wp.vec3),
    gt: wp.array(dtype=wp.vec3),
    num_object_points: int,
    loss: wp.array(dtype=wp.float32),
):
    # Calculate the smooth l1 loss modifed from fvcore.nn.smooth_l1_loss
    tid = wp.tid()
    pred_x = pred[tid][0]
    pred_y = pred[tid][1]
    pred_z = pred[tid][2]

    gt_x = gt[tid][0]
    gt_y = gt[tid][1]
    gt_z = gt[tid][2]

    dist_x = wp.abs(pred_x - gt_x)
    dist_y = wp.abs(pred_y - gt_y)
    dist_z = wp.abs(pred_z - gt_z)

    if dist_x < 1.0:
        temp_simple_loss_x = 0.5 * (dist_x**2.0)
    else:
        temp_simple_loss_x = dist_x - 0.5

    if dist_y < 1.0:
        temp_simple_loss_y = 0.5 * (dist_y**2.0)
    else:
        temp_simple_loss_y = dist_y - 0.5

    if dist_z < 1.0:
        temp_simple_loss_z = 0.5 * (dist_z**2.0)
    else:
        temp_simple_loss_z = dist_z - 0.5

    temp_simple_loss = temp_simple_loss_x + temp_simple_loss_y + temp_simple_loss_z

    average_factor = float(num_object_points) * 3.0

    final_simple_loss = temp_simple_loss / average_factor

    wp.atomic_add(loss, 0, final_simple_loss)


class SpringMassSystemWarp:
    def __init__(
        self,
        init_vertices,
        init_springs,
        init_rest_lengths,
        init_masses,
        dt,
        num_substeps,
        spring_Y,
        collide_elas,
        collide_fric,
        dashpot_damping,
        drag_damping,
        bend_damping=1.0,
        bend_stiffness = 1.5,  # Bending stiffness coefficient, optimized in log space.
        collide_object_elas=0.7,
        collide_object_fric=0.3,
        init_masks=None,
        collision_dist=0.02,
        init_velocities=None,
        num_object_points=None,
        num_surface_points=None,
        num_original_points=None,
        controller_points=None,
        reverse_z=False,
        spring_Y_min=1e3,
        spring_Y_max=1e5,
        bend_stiffness_min=0.0,  # Lower bound for bending stiffness.
        bend_stiffness_max=1e2,  # Upper bound for bending stiffness.
        gt_object_points=None,
        gt_object_visibilities=None,
        gt_object_motions_valid=None,
        self_collision=False,
        use_bending=True,
        disable_backward=False,
    ):
        import time
        logger.info(f"[SIMULATION]: Initialize the Spring-Mass System with {init_vertices.shape[0]} vertices, {init_springs.shape[0]} springs, {num_object_points} object points, {init_vertices.shape[0] - num_object_points} control points")
        self.device = cfg.device

        # FIXME: Compute initial spring angles automatically.
        start_time = time.time()
        init_rest_angles = compute_initial_spring_angles(
            init_vertices, init_springs, num_object_points, controller_points
        )
        end_time = time.time()
        logger.info(f"[SIMULATION]: Compute the initial spring angles in {end_time - start_time} seconds")
        
        # Record the parameters
        self.wp_init_vertices = wp.from_torch(
            init_vertices[:num_object_points].contiguous(),
            dtype=wp.vec3,
            requires_grad=False,
        )     # object points in warp tensor vec3->3D
        if init_velocities is None:
            self.wp_init_velocities = wp.zeros_like(
                self.wp_init_vertices, requires_grad=False
            ) # initial velocity is zero vec3->3D
        else:
            self.wp_init_velocities = wp.from_torch(
                init_velocities[:num_object_points].contiguous(),
                dtype=wp.vec3,
                requires_grad=False,
            )

        self.n_vertices = init_vertices.shape[0]
        self.n_springs = init_springs.shape[0]

        if controller_points is None:
            assert num_object_points == self.n_vertices
        else:
            assert (controller_points.shape[1] + num_object_points) == self.n_vertices

        self.num_object_points = num_object_points
        self.num_control_points = (
            controller_points.shape[1] if not controller_points is None else 0
        )
        # control points of hands
        self.controller_points = controller_points
        
        print(f"bending damping: {bend_damping}")
        print(f"bending stiffness: {bend_stiffness}")
        # Physical simulator parameters
        self.dt = dt
        self.num_substeps = num_substeps
        self.dashpot_damping = dashpot_damping
        self.drag_damping = drag_damping
        self.bend_damping = bend_damping
        self.use_bending = use_bending
        self.reverse_factor = 1.0 if not reverse_z else -1.0
        self.spring_Y_min = spring_Y_min
        self.spring_Y_max = spring_Y_max
        self.bend_stiffness_min = bend_stiffness_min
        self.bend_stiffness_max = bend_stiffness_max

        # Deal with the any collision detection
        # if it equals to 0, then collision only
        # happens between the object and the ground
        # if it equals to 1, then collision also happens
        # between the object points
        self.object_collision_flag = 0

        if init_masks is not None:
            if torch.unique(init_masks).shape[0] > 1:
                self.object_collision_flag = 1

        if self_collision:
            assert init_masks is None
            self.object_collision_flag = 1
            # Make all points as the collision points
            init_masks = torch.arange(
                self.n_vertices, 
                dtype=torch.int32, 
                device=self.device
            )

        if self.object_collision_flag:
            self.wp_masks = wp.from_torch(
                init_masks[:num_object_points].int(),
                dtype=wp.int32,
                requires_grad=False,
            )
            # discretize the space using hash grid
            # This is for calculating the potential collision points
            self.collision_grid = wp.HashGrid(128, 128, 128)
            self.collision_dist = collision_dist

            self.wp_collision_indices = wp.zeros(
                (self.wp_init_vertices.shape[0], 500),
                dtype=wp.int32,
                requires_grad=False,
            )
            self.wp_collision_number = wp.zeros(
                (self.wp_init_vertices.shape[0]), 
                dtype=wp.int32, requires_grad=False
            )

        # Initialize the GT for calculating losses
        self.gt_object_points = gt_object_points
        if cfg.data_type == "real":
            self.gt_object_visibilities = gt_object_visibilities.int()
            self.gt_object_motions_valid = gt_object_motions_valid.int()

        self.num_surface_points = num_surface_points
        self.num_original_points = num_original_points
        if num_original_points is None:
            self.num_original_points = self.num_object_points

        # Do some initialization to initialize the warp cuda graph
        self.wp_springs = wp.from_torch(
            init_springs, dtype=wp.vec2i, requires_grad=False
        )    # intialize the spring connection map: [n_points, n_points]
        self.wp_rest_lengths = wp.from_torch(
            init_rest_lengths, dtype=wp.float32, requires_grad=False
        )    # initial length of each centre points to neighbour points: [n_springs]
        # Initialize spring angle information [n_springs, 4].
        self.wp_spring_angle_info = wp.from_torch(
            init_rest_angles, dtype=wp.float32, requires_grad=False
        )    # Spring angle info: endpoint angles and reference-neighbor indices for each spring.
        self.wp_masses = wp.from_torch(
            init_masses[:num_object_points], 
            dtype=wp.float32, 
            requires_grad=False
        )   # mass of each point, [n_points]
        if cfg.data_type == "real":
            self.prev_acc = wp.zeros_like(
                self.wp_init_vertices, 
                requires_grad=False
                )    # initial prev_acc?
            self.acc_count = wp.zeros(
                1, dtype=wp.int32, requires_grad=False
                )

        self.wp_current_object_points = wp.from_torch(
            self.gt_object_points[1].clone(), 
            dtype=wp.vec3, 
            requires_grad=False
        )       # object points of the next frame (t+1)
        if cfg.data_type == "real":
            self.wp_current_object_visibilities = wp.from_torch(
                self.gt_object_visibilities[1].clone(),
                dtype=wp.int32,
                requires_grad=False,
            )   # visibility of object points of the next frame (t+1)
            self.wp_current_object_motions_valid = wp.from_torch(
                self.gt_object_motions_valid[0].clone(),
                dtype=wp.int32,
                requires_grad=False,
            )  # valid motion of object points between frame t and t+1
            self.num_valid_visibilities = int(
                self.gt_object_visibilities[1].sum()
                )   # number of visible object points in the next frame (t+1)
            self.num_valid_motions = int(
                self.gt_object_motions_valid[0].sum()
                )   # number of object points with valid motions between frame t and t+1
 
            self.wp_original_control_point = wp.from_torch(
                self.controller_points[0].clone(), 
                dtype=wp.vec3, 
                requires_grad=False
            )   # control points of hands of the current frame (t)
            self.wp_target_control_point = wp.from_torch(
                self.controller_points[1].clone(), 
                dtype=wp.vec3, requires_grad=False
            )   # control points of hands of the next frame (t+1)

            self.chamfer_loss = wp.zeros(1, dtype=wp.float32, requires_grad=True)
            self.track_loss = wp.zeros(1, dtype=wp.float32, requires_grad=True)
            self.acc_loss = wp.zeros(1, dtype=wp.float32, requires_grad=True)
            
        self.loss = wp.zeros(1, dtype=wp.float32, requires_grad=True)

        # Initialize the warp parameters
        self.wp_states = []
        for i in range(self.num_substeps + 1):
            # initialize every single state in each substep
            # The initialized velocities and other parameters
            # are equal to 0 for optimization
            state = State(self.wp_init_velocities, self.num_control_points)
            self.wp_states.append(state)
        if cfg.data_type == "real":
            self.distance_matrix = wp.zeros(
                (self.num_original_points, self.num_surface_points), 
                requires_grad=False
            )
            self.neigh_indices = wp.zeros(
                (self.num_original_points), 
                dtype=wp.int32, requires_grad=False
            )

        # Parameter to be optimized
        self.wp_spring_Y = wp.from_torch(
            torch.log(
                torch.tensor(
                    spring_Y, dtype=torch.float32, 
                    device=self.device)
                    )
            * torch.ones(
                self.n_springs, 
                dtype=torch.float32, 
                device=self.device
                ),
            requires_grad=True,
        )
        # Initialize bending stiffness coefficients for optimization in log space.
        self.wp_bending_stiffness = wp.from_torch(
            torch.log(
                torch.tensor(
                    bend_stiffness, dtype=torch.float32,
                    device=self.device)
                )
            * torch.ones(
                self.num_object_points,  # Object points only
                dtype=torch.float32,
                device=self.device
                ),
            requires_grad=True,
        )
        self.wp_collide_elas = wp.from_torch(
            torch.tensor([collide_elas], dtype=torch.float32, device=self.device),
            requires_grad=cfg.collision_learn,
        )
        self.wp_collide_fric = wp.from_torch(
            torch.tensor([collide_fric], dtype=torch.float32, device=self.device),
            requires_grad=cfg.collision_learn,
        )
        self.wp_collide_object_elas = wp.from_torch(
            torch.tensor(
                [collide_object_elas], dtype=torch.float32, device=self.device
            ),
            requires_grad=cfg.collision_learn,
        )
        self.wp_collide_object_fric = wp.from_torch(
            torch.tensor(
                [collide_object_fric], dtype=torch.float32, device=self.device
            ),
            requires_grad=cfg.collision_learn,
        )

        # Create the CUDA graph to acclerate
        if cfg.use_graph:
            if cfg.data_type == "real":
                if not disable_backward:
                    with wp.ScopedCapture() as capture:
                        self.tape = wp.Tape()
                        with self.tape:
                            self.step()
                            self.calculate_loss()
                        self.tape.backward(self.loss)
                else:
                    with wp.ScopedCapture() as capture:
                        self.step()
                        self.calculate_loss()
                self.graph = capture.graph
            elif cfg.data_type == "synthetic":
                if not disable_backward:
                    # For synthetic data, we compute simple loss
                    with wp.ScopedCapture() as capture:
                        self.tape = wp.Tape()
                        with self.tape:
                            self.step()
                            self.calculate_simple_loss()
                        self.tape.backward(self.loss)
                else:
                    with wp.ScopedCapture() as capture:
                        self.step()
                        self.calculate_simple_loss()
                self.graph = capture.graph
            else:
                raise NotImplementedError

            with wp.ScopedCapture() as forward_capture:
                self.step()
            self.forward_graph = forward_capture.graph
        else:
            self.tape = wp.Tape()

    def set_controller_target(self, frame_idx, pure_inference=False):
        if self.controller_points is not None:
            # Set the controller points
            wp.launch(
                copy_vec3,       # control points of hands of the current frame (t)
                dim=self.num_control_points,
                inputs=[self.controller_points[frame_idx - 1]],
                outputs=[self.wp_original_control_point],
            )
            wp.launch(
                copy_vec3,      # control points of hands of the next frame (t+1)
                dim=self.num_control_points,
                inputs=[self.controller_points[frame_idx]],
                outputs=[self.wp_target_control_point],
            )

        if not pure_inference:
            # Set the target points
            wp.launch(
                copy_vec3,      # GT object points of the next frame (t+1)
                dim=self.num_original_points,
                inputs=[self.gt_object_points[frame_idx]],
                outputs=[self.wp_current_object_points],
            )

            if cfg.data_type == "real":
                wp.launch(  
                    copy_int,   # GT object visibilities of the next frame (t+1)
                    dim=self.num_original_points,
                    inputs=[self.gt_object_visibilities[frame_idx]],
                    outputs=[self.wp_current_object_visibilities],
                )
                wp.launch(
                    copy_int,   # GT object motions valid between frame t and t+1
                    dim=self.num_original_points,
                    inputs=[self.gt_object_motions_valid[frame_idx - 1]],
                    outputs=[self.wp_current_object_motions_valid],
                )

                self.num_valid_visibilities = int(
                    self.gt_object_visibilities[frame_idx].sum()
                )           # number of visible object points in the next frame (t+1)
                self.num_valid_motions = int(
                    self.gt_object_motions_valid[frame_idx - 1].sum()
                )           # number of object points with valid motions between frame t and t+1

    def set_controller_interactive(
        self, last_controller_interactive, controller_interactive
    ):
        # Set the controller points
        wp.launch(
            copy_vec3,
            dim=self.num_control_points,
            inputs=[last_controller_interactive],
            outputs=[self.wp_original_control_point],
        )
        wp.launch(
            copy_vec3,
            dim=self.num_control_points,
            inputs=[controller_interactive],
            outputs=[self.wp_target_control_point],
        )

    def set_init_state(self, wp_x, wp_v, pure_inference=False):
        # Detach and clone and set requires_grad=True
        assert (
            self.num_object_points == wp_x.shape[0]
            and self.num_object_points == self.wp_states[0].wp_x.shape[0]
        )

        if not pure_inference:
            wp.launch(
                copy_vec3,
                dim=self.num_object_points,
                inputs=[wp.clone(wp_x, requires_grad=False)],
                outputs=[self.wp_states[0].wp_x],
            )
            wp.launch(
                copy_vec3,
                dim=self.num_object_points,
                inputs=[wp.clone(wp_v, requires_grad=False)],
                outputs=[self.wp_states[0].wp_v],
            )
        else:
            wp.launch(
                copy_vec3,
                dim=self.num_object_points,
                inputs=[wp_x],
                outputs=[self.wp_states[0].wp_x],
            )
            wp.launch(
                copy_vec3,
                dim=self.num_object_points,
                inputs=[wp_v],
                outputs=[self.wp_states[0].wp_v],
            )

    def set_acc_count(self, acc_count):
        if acc_count:
            input = 1
        else:
            input = 0
        wp.launch(
            set_int,
            dim=1,
            inputs=[input],
            outputs=[self.acc_count],
        )

    def update_acc(self):
        wp.launch(
            update_acc,
            dim=self.num_object_points,
            inputs=[
                wp.clone(self.wp_states[0].wp_v, requires_grad=False),
                wp.clone(self.wp_states[-1].wp_v, requires_grad=False),
            ],
            outputs=[self.prev_acc],
        )

    def update_collision_graph(self):
        assert self.object_collision_flag
        # insert the object points into the hash grid(e.g., 128*128*128 voxels)
        # the Hash grid enable fast query of potential collision points
        self.collision_grid.build(
            self.wp_states[0].wp_x,   # initialize on the movement of frame t
            self.collision_dist * 5.0 # 5.0 is the scale factor of the voxel size
            )
        self.wp_collision_number.zero_()
        wp.launch(
            update_potential_collision,
            dim=self.num_object_points,
            inputs=[
                self.wp_states[0].wp_x,  # not sure why use the initial state
                self.wp_masks,
                self.collision_dist,
                self.collision_grid.id,
            ],
            outputs=[self.wp_collision_indices, self.wp_collision_number],
        )

    def step(self):
        for i in range(self.num_substeps):
            self.wp_states[i].clear_forces()
            if not self.controller_points is None:
                # Set the control point of hands by interpolating between 
                # the original and target control points 
                # (e.g, t to t+1 frame)
                wp.launch(
                    set_control_points,
                    dim=self.num_control_points,
                    inputs=[
                        self.num_substeps,
                        self.wp_original_control_point,
                        self.wp_target_control_point,
                        i,
                    ],
                    outputs=[self.wp_states[i].wp_control_x],
                )

            # Calculate the spring forces
            wp.launch(
                kernel=eval_springs,
                dim=self.n_springs,
                inputs=[
                    self.wp_states[i].wp_x,
                    self.wp_states[i].wp_v,
                    self.wp_states[i].wp_control_x,
                    self.wp_states[i].wp_control_v,
                    self.num_object_points,
                    self.wp_springs,
                    self.wp_rest_lengths,
                    self.wp_spring_Y,
                    self.dashpot_damping,
                    self.spring_Y_min,
                    self.spring_Y_max,
                ],
                outputs=[self.wp_states[i].wp_vertice_forces],
            )   # the force for each vertex is updated based on the spring elongation

            if self.use_bending:
                wp.launch(
                    kernel=eval_bending_forces,
                    dim=self.n_springs,  # Parallelize over springs.
                    inputs=[
                        self.wp_states[i].wp_x,
                        self.wp_states[i].wp_v,
                        self.wp_states[i].wp_control_x,
                        self.wp_states[i].wp_control_v,
                        self.num_object_points,
                        self.wp_springs,
                        self.wp_spring_angle_info,  # Use the new data layout.
                        self.wp_bending_stiffness,
                        self.bend_damping,
                        self.bend_stiffness_min,
                        self.bend_stiffness_max,
                    ],
                    outputs=[self.wp_states[i].wp_vertice_forces],  # Accumulate into the same force array.
                )

            # Collision handling between object points, 
            # and between object and ground
            if self.object_collision_flag:
                output_v = self.wp_states[i].wp_v_before_collision
            else:
                output_v = self.wp_states[i].wp_v_before_ground

            # Update the output_v using the forces calculated above
            # in self.wp_states[i].wp_vertice_forces
            wp.launch(
                kernel=update_vel_from_force,
                dim=self.num_object_points,
                inputs=[
                    self.wp_states[i].wp_v,
                    self.wp_states[i].wp_vertice_forces,
                    self.wp_masses,
                    self.dt,
                    self.drag_damping,
                    self.reverse_factor,
                ],
                outputs=[output_v],
            )

            if self.object_collision_flag:
                # Update the wp_v_before_ground based on the collision handling
                wp.launch(
                    kernel=object_collision,
                    dim=self.num_object_points,
                    inputs=[
                        self.wp_states[i].wp_x,
                        # wp_v_before_collision for object-object collision
                        self.wp_states[i].wp_v_before_collision,
                        self.wp_masses,
                        self.wp_masks,
                        self.wp_collide_object_elas,
                        self.wp_collide_object_fric,
                        self.collision_dist,
                        self.wp_collision_indices,
                        self.wp_collision_number,
                    ],
                    outputs=[self.wp_states[i].wp_v_before_ground],
                )

            # Update the x and v
            wp.launch(
                kernel=integrate_ground_collision,
                dim=self.num_object_points,
                inputs=[
                    self.wp_states[i].wp_x,
                    self.wp_states[i].wp_v_before_ground,
                    self.wp_collide_elas,
                    self.wp_collide_fric,
                    self.dt,
                    self.reverse_factor,
                ],
                outputs=[self.wp_states[i + 1].wp_x, self.wp_states[i + 1].wp_v],
            )

    def calculate_loss(self):
        # Compute the chamfer loss
        # Precompute the distances matrix for the chamfer loss
        wp.launch(
            compute_distances,
            dim=(self.num_original_points, self.num_surface_points),
            inputs=[
                self.wp_states[-1].wp_x,
                self.wp_current_object_points,
                self.wp_current_object_visibilities,
            ],
            outputs=[self.distance_matrix],
        )

        wp.launch(
            compute_neigh_indices,
            dim=self.num_original_points,
            inputs=[self.distance_matrix],
            outputs=[self.neigh_indices],
        )

        wp.launch(
            compute_chamfer_loss,
            dim=self.num_original_points,
            inputs=[
                self.wp_states[-1].wp_x,               # Predicted positions
                self.wp_current_object_points,         # GT positions
                self.wp_current_object_visibilities,   # Visibility mask
                self.num_valid_visibilities,           # Number of visible points
                self.neigh_indices,                    # Closest point correspondences
                cfg.chamfer_weight,                    # Loss weight hyperparameter
            ],
            outputs=[self.chamfer_loss],
        )

        # Compute the tracking loss
        wp.launch(
            compute_track_loss,
            dim=self.num_original_points,
            inputs=[
                self.wp_states[-1].wp_x,       # velocity in frame t
                self.wp_current_object_points, # velocity in frame t+1   
                self.wp_current_object_motions_valid,
                self.num_valid_motions,
                cfg.track_weight,
            ],
            outputs=[self.track_loss],
        )

        wp.launch(
            compute_acc_loss,
            dim=self.num_object_points,
            inputs=[
                self.wp_states[0].wp_v,
                self.wp_states[-1].wp_v,
                self.prev_acc,
                self.num_object_points,
                self.acc_count,
                cfg.acc_weight,
            ],
            outputs=[self.acc_loss],
        )

        wp.launch(
            compute_final_loss,
            dim=1,
            inputs=[self.chamfer_loss, self.track_loss, self.acc_loss],
            outputs=[self.loss],
        )

    def calculate_simple_loss(self):
        wp.launch(
            compute_simple_loss,
            dim=self.num_object_points,
            inputs=[
                self.wp_states[-1].wp_x,
                self.wp_current_object_points,
                self.num_object_points,
            ],
            outputs=[self.loss],
        )

    def clear_loss(self):
        if cfg.data_type == "real":
            self.distance_matrix.zero_()
            self.neigh_indices.zero_()
            self.chamfer_loss.zero_()
            self.track_loss.zero_()
            self.acc_loss.zero_()
        self.loss.zero_()

    # Functions used to load the parmeters
    def set_spring_Y(self, spring_Y):
        # assert spring_Y.shape[0] == self.n_springs
        wp.launch(
            copy_float,
            dim=self.n_springs,
            inputs=[spring_Y],
            outputs=[self.wp_spring_Y],
        )

    def set_bend_stiffness(self, bend_stiffness):
        # assert bend_stiffness.shape[0] == self.num_object_points
        wp.launch(
            copy_float,
            dim=self.num_object_points,
            inputs=[bend_stiffness],
            outputs=[self.wp_bending_stiffness],
        )

    def set_collide(self, collide_elas, collide_fric):
        wp.launch(
            copy_float,
            dim=1,
            inputs=[collide_elas],
            outputs=[self.wp_collide_elas],
        )
        wp.launch(
            copy_float,
            dim=1,
            inputs=[collide_fric],
            outputs=[self.wp_collide_fric],
        )

    def set_collide_object(self, collide_object_elas, collide_object_fric):
        wp.launch(
            copy_float,
            dim=1,
            inputs=[collide_object_elas],
            outputs=[self.wp_collide_object_elas],
        )
        wp.launch(
            copy_float,
            dim=1,
            inputs=[collide_object_fric],
            outputs=[self.wp_collide_object_fric],
        )
