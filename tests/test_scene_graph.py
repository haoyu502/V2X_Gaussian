import copy
import math
import unittest
from types import SimpleNamespace
import numpy as np
import torch
from scene.scene_graph import (ActorGaussianModel, ActorPose, BackgroundGaussianModel,
                               GaussianSceneGraph, TrackletManager, quaternion_to_matrix)
from scene.scene_graph.actor_model import transform_actor_gaussians
from scene.scene_graph.timestamps import camera_metadata, timestamp_ns_from_filename
from scene.scene_graph.preprocessing import read_pcd_xyz, split_points, voxel_sample
from utils.graphics_utils import getProjectionMatrix


def make_graph(sh_degree=0):
    bg = BackgroundGaussianModel([[0., -0.8, 5.]], [[0.1, 0.2, 0.8]], sh_degree, 0.2)
    actor = ActorGaussianModel([[-0.2, 0., 0.], [0.2, 0., 0.]],
                              [[0.9, 0.2, 0.1], [0.2, 0.9, 0.1]], sh_degree, 0.16, 0.6)
    poses = TrackletManager({"car": ActorPose([0., 1.], [[-0.4, 0, 4], [0.4, 0, 4]],
                                             [[1, 0, 0, 0], [1, 0, 0, 0]], 2)}, 1688625741000000000)
    return GaussianSceneGraph(bg, {"car": actor}, poses)


def make_camera(timestamp, center_x=0):
    view = torch.eye(4)
    view[3, 0] = -center_x
    projection = getProjectionMatrix(0.01, 100., 1., 1.).T
    return SimpleNamespace(timestamp_sec=timestamp, time_origin_ns=1688625741000000000,
                           image_width=64, image_height=64, FoVx=1., FoVy=1.,
                           world_view_transform=view, full_proj_transform=view @ projection,
                           camera_center=torch.tensor([center_x, 0., 0.]))


class SceneGraphMathTest(unittest.TestCase):
    def test_per_frame_split_uses_rotated_boxes_and_removes_excluded_objects(self):
        objects = {
            "car": {"position": [1, 0, 0], "size": [2, 0.2, 0.2],
                    "rotation_wxyz": [math.sqrt(0.5), 0, 0, math.sqrt(0.5)]},
            "pedestrian": {"position": [-1, 0, 0], "size": [0.5, 0.5, 2], "rotation_wxyz": [1, 0, 0, 0]},
        }
        points = np.array([[1, 0.8, 0], [-1, 0, 0], [1.8, 0, 0], [5, 5, 5]], dtype=np.float32)
        background, actors = split_points(points, objects, margin=0)
        np.testing.assert_allclose(actors["car"], [[0.8, 0, 0]], atol=1e-6)
        np.testing.assert_allclose(actors["pedestrian"], [[0, 0, 0]])
        np.testing.assert_allclose(background, points[[2, 3]])
        self.assertEqual(len(background)+sum(map(len, actors.values())), len(points))

    def test_pcd_xyz_filters_invalid_and_zero_sensor_returns(self):
        header = ("VERSION 0.7\nFIELDS x y z intensity\nSIZE 4 4 4 4\nTYPE F F F F\n"
                  "COUNT 1 1 1 1\nWIDTH 3\nHEIGHT 1\nPOINTS 3\nDATA ")
        ascii_data = (header+"ascii\n0 0 0 1\n1 2 3 4\nnan 1 2 3\n").encode()
        np.testing.assert_allclose(read_pcd_xyz(ascii_data), [[1, 2, 3]])
        binary_data = (header+"binary\n").encode()+np.array([[0, 0, 0, 1], [1, 2, 3, 4], [np.nan, 1, 2, 3]], dtype="<f4").tobytes()
        np.testing.assert_allclose(read_pcd_xyz(binary_data), [[1, 2, 3]])
        self.assertEqual(len(voxel_sample([[0, 0, 0], [0.01, 0, 0], [1, 1, 1]], 0.1, 10)), 2)

    def test_integer_timestamps_preserve_offsets(self):
        origin = 1688625741000000000
        meta = camera_metadata("1688625741_052165680_vehicle_camera_basler_16mm.jpg", origin)
        self.assertEqual(meta["timestamp_ns"], origin+52165680)
        self.assertAlmostEqual(meta["timestamp_sec"], 0.05216568, places=12)
        self.assertEqual(meta["agent"], "ego")
        self.assertEqual(timestamp_ns_from_filename("1688625741_037483960_sensor.pcd"), origin+37483960)
        with self.assertRaises(ValueError):
            timestamp_ns_from_filename("000001.jpg")

    def test_pose_translation_and_shortest_rotation(self):
        # Crossing +170 -> -170 degrees must pass through 180, not zero.
        a = math.radians(170)/2
        pose = ActorPose([0., 1.], [[0, 0, 0], [2, 0, 0]],
                         [[math.cos(a), 0, 0, math.sin(a)],
                          [math.cos(a), 0, 0, -math.sin(a)]], 2)
        rotation, position, _ = pose(0.5)
        torch.testing.assert_close(position, torch.tensor([1., 0., 0.]))
        torch.testing.assert_close(rotation @ torch.tensor([1., 0., 0.]), torch.tensor([-1., 0., 0.]))
        self.assertFalse(pose.is_active(-0.01))
        with self.assertRaises(ValueError):
            pose(1.1)

    def test_gaps_and_exact_observations(self):
        pose = ActorPose([0., 1.], [[0, 0, 0], [1, 0, 0]], [[1, 0, 0, 0]]*2, 0.2)
        self.assertTrue(pose.is_active(0))
        self.assertTrue(pose.is_active(1))
        self.assertFalse(pose.is_active(0.5))
        with self.assertRaises(ValueError):
            ActorPose([1, 1], [[0, 0, 0]]*2, [[1, 0, 0, 0]]*2)

    def test_rigid_rotation_preserves_anisotropic_covariance(self):
        actor = make_graph().actors["car"]
        with torch.no_grad():
            actor.log_scaling[0] = torch.tensor([0.1, 0.2, 0.4]).log()
            actor.rotation_raw[0] = torch.tensor([math.cos(0.2), math.sin(0.2), 0., 0.])
        pack = actor.get_gaussian_pack()
        q = torch.tensor([math.sqrt(0.5), 0., 0., math.sqrt(0.5)])
        rotation = quaternion_to_matrix(q)
        transformed = transform_actor_gaussians(pack, rotation, torch.tensor([1., 2., 3.]), q)
        local_rotation = quaternion_to_matrix(pack.rotation)
        world_rotation = quaternion_to_matrix(transformed.rotation)
        local_cov = local_rotation @ torch.diag_embed(pack.scaling.square()) @ local_rotation.transpose(-1, -2)
        world_cov = world_rotation @ torch.diag_embed(pack.scaling.square()) @ world_rotation.transpose(-1, -2)
        torch.testing.assert_close(world_cov, rotation @ local_cov @ rotation.T)
        torch.testing.assert_close(transformed.xyz, pack.xyz @ rotation.T+torch.tensor([1., 2., 3.]))

    def test_shared_nodes_and_gradient_slices(self):
        graph = make_graph()
        first, last = graph.compose(0), graph.compose(1)
        self.assertEqual(first.node_slices["car"], slice(1, 3))
        torch.testing.assert_close(first.pack.xyz[0], last.pack.xyz[0])
        torch.testing.assert_close(last.pack.xyz[1:]-first.pack.xyz[1:], torch.tensor([[0.8, 0, 0]]*2))
        (first.pack.xyz.sum()+last.pack.xyz.sum()).backward()
        torch.testing.assert_close(graph.actors["car"].xyz.grad, torch.full((2, 3), 2.))
        self.assertEqual(set(graph.compose(-1).node_slices), {"background"})

    def test_actor_sh_rotates_in_canonical_frame(self):
        graph = make_graph(1)
        with torch.no_grad():
            graph.actors["car"].features[:, 3] = 0.7
        first = graph.compose(0)
        # Rotate the whole actor, its translation and camera by the same R.
        q = torch.tensor([math.sqrt(0.5), 0., 0., math.sqrt(0.5)])
        rotation = quaternion_to_matrix(q)
        with torch.no_grad():
            graph.tracklets.poses["car"].rotations_wxyz[:] = q
            graph.tracklets.poses["car"].positions[:] = graph.tracklets.poses["car"].positions @ rotation.T
        second = graph.compose(0)
        camera = torch.tensor([1., 2., -1.])
        torch.testing.assert_close(graph.colors(first, camera)[1:], graph.colors(second, camera @ rotation.T)[1:])

    def test_state_dict_round_trip(self):
        graph = make_graph()
        state = copy.deepcopy(graph.state_dict())
        with torch.no_grad():
            graph.actors["car"].xyz.add_(2)
        graph.load_state_dict(state)
        torch.testing.assert_close(graph.compose(0.5).pack.xyz[1:], torch.tensor([[-0.2, 0., 4.], [0.2, 0., 4.]]))


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class SceneGraphRasterizerTest(unittest.TestCase):
    def test_color_gradient_matches_finite_difference(self):
        from gaussian_renderer.scene_graph_renderer import render_scene_graph
        graph = make_graph().cuda()
        pipe, bg, camera = SimpleNamespace(debug=False), torch.zeros(3, device="cuda"), make_camera(0.5)
        parameter = graph.actors["car"].features
        render_scene_graph(camera, graph, pipe, bg)["render"].square().mean().backward()
        analytical = float(parameter.grad[0, 0, 0])
        original = float(parameter[0, 0, 0])
        epsilon = 1e-3
        values = []
        with torch.no_grad():
            for delta in (epsilon, -epsilon):
                parameter[0, 0, 0] = original+delta
                values.append(float(render_scene_graph(camera, graph, pipe, bg)["render"].square().mean()))
            parameter[0, 0, 0] = original
        numerical = (values[0]-values[1])/(2*epsilon)
        self.assertGreater(abs(analytical), 1e-8)
        self.assertAlmostEqual(analytical, numerical, delta=max(1e-7, abs(numerical)*0.02))

    def test_three_cameras_supervise_one_actor(self):
        from gaussian_renderer.scene_graph_renderer import render_scene_graph
        graph = make_graph(1).cuda()
        pipe, bg = SimpleNamespace(debug=False), torch.zeros(3, device="cuda")
        gradients = []
        for timestamp, center in [(0.4, 0.), (0.45, -0.5), (0.5, 0.5)]:
            graph.zero_grad(set_to_none=True)
            result = render_scene_graph(make_camera(timestamp, center), graph, pipe, bg)
            self.assertEqual(result["render"].shape, (3, 64, 64))
            self.assertTrue(torch.isfinite(result["depth"]).all())
            self.assertTrue(result["visibility_filter"][result["node_slices"]["car"]].any())
            result["render"].square().mean().backward()
            self.assertIsNotNone(result["viewspace_points"].grad)
            for name in ("xyz", "features", "log_scaling", "opacity_logits", "rotation_raw"):
                grad = getattr(graph.actors["car"], name).grad
                self.assertIsNotNone(grad, name)
                self.assertTrue(torch.isfinite(grad).all(), name)
            gradients.append(graph.actors["car"].features.grad.clone())
            self.assertGreater(float(gradients[-1].abs().sum()), 0)
        graph.zero_grad(set_to_none=True)
        for timestamp, center in [(0.4, 0.), (0.45, -0.5), (0.5, 0.5)]:
            image = render_scene_graph(make_camera(timestamp, center), graph, pipe, bg)["render"]
            (image.square().mean()/3).backward()
        torch.testing.assert_close(graph.actors["car"].features.grad, torch.stack(gradients).mean(0), rtol=1e-4, atol=1e-8)

    def test_rejects_mixed_time_origins(self):
        from gaussian_renderer.scene_graph_renderer import render_scene_graph
        camera = make_camera(0.5)
        camera.time_origin_ns += 1
        with self.assertRaises(ValueError):
            render_scene_graph(camera, make_graph().cuda(), SimpleNamespace(debug=False), torch.zeros(3, device="cuda"))


if __name__ == "__main__":
    unittest.main()
