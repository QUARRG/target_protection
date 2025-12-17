        # prev_leader_phase, prev_ego_phase, prev_follower_phase = phases
        # Re = build_Re(self.embedding_fn, prev_ego_phase)
        # Rc = build_Rc(prev_ego_phase)
        # p = Rc.T @ Re.T @ current_pose
        # self.radius  = np.sqrt(p[0]**2 + p[1]**2 + p[2]**2)

        # pose = Re.T @ current_pose
        # current_ego_phase = wrap_to_2pi(np.arctan2(pose[1], pose[0]))
        # # Rc = build_Rc(current_ego_phase)

        # # curr_ego_phase = np.arctan2(p[1], p[0])

        # omega, gain = phase_controller(current_ego_phase, prev_leader_phase, prev_follower_phase, self.omega_nominal, self.k_phi)
        # # Update phase
        # des_ego_pose_2D = np.array([self.radius_nominal*np.cos(current_ego_phase),self.radius_nominal*np.sin(current_ego_phase), 0])
        # desired_ego_pose = exp_SO3(np.asarray([0., 0., omega * self.dt])) @ des_ego_pose_2D
        # desired_ego_phase = wrap_to_2pi(np.arctan2(desired_ego_pose[1], desired_ego_pose[0]))
        # des_Re = build_Re(self.embedding_fn, desired_ego_phase)
        # desired_ego_pose_3D = des_Re@desired_ego_pose
        
        
        # # Publish predicted pose, phase and controller gain
        # # current_pose_msg, desired_pose_msg, phase_msg, radius_msg = self.build_pose_phase_msgs()
        # phase_msg_test = Float32()
        # phase_msg_test.data = current_ego_phase
        # desired_pose_msg = PoseWithCovarianceStamped()
        # desired_pose_msg.header.frame_id = self.frame_id
        # desired_pose_msg.header.stamp = self.node.get_clock().now().to_msg()
        # desired_pose_msg.pose.pose.position = Point(x=desired_ego_pose_3D[0], y=desired_ego_pose_3D[1], z=desired_ego_pose_3D[2])
        # desired_pose_msg.pose.pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
        # radius_msg = Float32()
        # radius_msg.data = self.radius
        # # self.pub_pose.publish(current_pose_msg)
        # self.pub_phase.publish(phase_msg_test)
        # self.pub_radius.publish(radius_msg)

        # omega_msg = Float32()
        # omega_msg.data = omega
        # self.pub_omega.publish(omega_msg)

        # gain_msg = Float32()
        # gain_msg.data = gain
        # self.pub_gain.publish(gain_msg)