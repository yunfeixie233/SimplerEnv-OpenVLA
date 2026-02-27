"""
Evaluate a model on ManiSkill2 environment.
"""

import json
import os

import numpy as np
from PIL import Image
from transforms3d.euler import quat2euler

from simpler_env.utils.env.env_builder import (
    build_maniskill2_env,
    get_robot_control_mode,
)
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
from simpler_env.utils.visualization import write_interval_video, write_video


def run_maniskill2_eval_single_episode(
    model,
    ckpt_path,
    robot_name,
    env_name,
    scene_name,
    robot_init_x,
    robot_init_y,
    robot_init_quat,
    control_mode,
    obj_init_x=None,
    obj_init_y=None,
    obj_episode_id=None,
    additional_env_build_kwargs=None,
    rgb_overlay_path=None,
    obs_camera_name=None,
    control_freq=3,
    sim_freq=513,
    max_episode_steps=80,
    instruction=None,
    enable_raytracing=False,
    additional_env_save_tags=None,
    logging_dir="./results",
    save_rollout_dir=None,
):
    if additional_env_build_kwargs is None:
        additional_env_build_kwargs = {}

    # Create environment
    kwargs = dict(
        obs_mode="rgbd",
        robot=robot_name,
        sim_freq=sim_freq,
        control_mode=control_mode,
        control_freq=control_freq,
        max_episode_steps=max_episode_steps,
        scene_name=scene_name,
        camera_cfgs={"add_segmentation": True},
        rgb_overlay_path=rgb_overlay_path,
    )
    if enable_raytracing:
        ray_tracing_dict = {"shader_dir": "rt"}
        ray_tracing_dict.update(additional_env_build_kwargs)
        # put raytracing dict keys before other keys for compatibility with existing result naming and metric calculation
        additional_env_build_kwargs = ray_tracing_dict
    env = build_maniskill2_env(
        env_name,
        **additional_env_build_kwargs,
        **kwargs,
    )
    # __import__('ipdb').set_trace()
    # initialize environment
    env_reset_options = {
        "robot_init_options": {
            "init_xy": np.array([robot_init_x, robot_init_y]),
            "init_rot_quat": robot_init_quat,
        }
    }
    if obj_init_x is not None:
        assert obj_init_y is not None
        obj_variation_mode = "xy"
        env_reset_options["obj_init_options"] = {
            "init_xy": np.array([obj_init_x, obj_init_y]),
        }
    else:
        assert obj_episode_id is not None
        obj_variation_mode = "episode"
        env_reset_options["obj_init_options"] = {
            "episode_id": obj_episode_id,
        }
    obs, _ = env.reset(options=env_reset_options)
    # for long-horizon environments, we check if the current subtask is the final subtask
    is_final_subtask = env.unwrapped.is_final_subtask()

    # Obtain language instruction
    if instruction is not None:
        task_description = instruction
    else:
        # get default language instruction
        task_description = env.unwrapped.get_language_instruction()
    print(task_description)

    # Initialize logging
    image = get_image_from_maniskill2_obs_dict(env, obs, camera_name=obs_camera_name)
    images = [image]
    predicted_actions = []
    predicted_terminated, done, truncated = False, False, False

    # Initialize model
    model.reset(task_description)

    timestep = 0
    success = "failure"

    # Rollout saving state
    rollout_images = []
    rollout_eef_poses = []
    rollout_actions = []
    rollout_task_descs = []
    rollout_done = False

    # Step the environment
    task_descriptions = []
    while not (predicted_terminated or truncated):
        # step the model; "raw_action" is raw model action output; "action" is the processed action to be sent into maniskill env
        if "extra" in obs and "tcp_pose" in obs["extra"]:
            tcp_pose = obs["extra"]["tcp_pose"]
            gripper_qpos = obs["agent"]["qpos"][-2:]
            gripper_openness = np.mean(gripper_qpos) / 0.037
            eef_pos = np.concatenate([tcp_pose, [gripper_openness]])
        else:
            eef_pos = obs["agent"]["eef_pos"]
        raw_action, action = model.step(image, task_description, eef_pos=eef_pos)
        predicted_actions.append(raw_action)
        predicted_terminated = bool(action["terminate_episode"][0] > 0)
        if predicted_terminated:
            if not is_final_subtask:
                # advance the environment to the next subtask
                predicted_terminated = False
                env.unwrapped.advance_to_next_subtask()

        # Accumulate rollout data at the decision point (before env.step)
        if save_rollout_dir is not None and not rollout_done:
            env_action_7d = np.concatenate(
                [action["world_vector"], action["rot_axangle"], action["gripper"]]
            )
            rollout_images.append(image)
            rollout_eef_poses.append(eef_pos.copy())
            rollout_actions.append(env_action_7d)
            rollout_task_descs.append(task_description)

        # step the environment
        obs, reward, done, truncated, info = env.step(
            np.concatenate(
                [action["world_vector"], action["rot_axangle"], action["gripper"]]
            ),
        )

        success = "success" if done else "failure"
        if done and not rollout_done:
            rollout_done = True
        new_task_description = env.unwrapped.get_language_instruction()
        if new_task_description != task_description:
            task_description = new_task_description
            print(task_description)
        is_final_subtask = env.unwrapped.is_final_subtask()

        print(timestep, info)

        image = get_image_from_maniskill2_obs_dict(
            env, obs, camera_name=obs_camera_name
        )
        images.append(image)
        task_descriptions.append(task_description)
        timestep += 1

    episode_stats = info.get("episode_stats", {})

    # Save rollout data
    if save_rollout_dir is not None and len(rollout_images) > 0:
        ep_id = obj_episode_id if obj_episode_id is not None else f"{obj_init_x}_{obj_init_y}"
        ep_dir = os.path.join(save_rollout_dir, env_name, f"episode_{ep_id:03d}" if isinstance(ep_id, int) else f"episode_{ep_id}")
        img_dir = os.path.join(ep_dir, "images")
        os.makedirs(img_dir, exist_ok=True)
        for i, img in enumerate(rollout_images):
            Image.fromarray(img).save(os.path.join(img_dir, f"step_{i:04d}.png"))
        np.save(os.path.join(ep_dir, "eef_poses.npy"), np.array(rollout_eef_poses, dtype=np.float32))
        np.save(os.path.join(ep_dir, "env_actions.npy"), np.array(rollout_actions, dtype=np.float32))
        meta = {
            "env_name": env_name,
            "task_description": rollout_task_descs[0] if rollout_task_descs else task_description,
            "num_steps": len(rollout_images),
            "success": success == "success",
            "episode_id": int(ep_id) if isinstance(ep_id, int) else str(ep_id),
        }
        with open(os.path.join(ep_dir, "metadata.json"), "w") as f:
            json.dump(meta, f, indent=2)
        print(f"[rollout] Saved {len(rollout_images)} transitions to {ep_dir}")

    # save video
    env_save_name = env_name

    for k, v in additional_env_build_kwargs.items():
        env_save_name = env_save_name + f"_{k}_{v}"
    if additional_env_save_tags is not None:
        env_save_name = env_save_name + f"_{additional_env_save_tags}"
    ckpt_path_basename = ckpt_path if ckpt_path[-1] != "/" else ckpt_path[:-1]
    ckpt_path_basename = ckpt_path_basename.split("/")[-1]
    if obj_variation_mode == "xy":
        video_name = f"{success}_obj_{obj_init_x}_{obj_init_y}"
    elif obj_variation_mode == "episode":
        video_name = f"{success}_obj_episode_{obj_episode_id}"
    for k, v in episode_stats.items():
        video_name = video_name + f"_{k}_{v}"
    video_name = video_name + ".mp4"
    if rgb_overlay_path is not None:
        rgb_overlay_path_str = os.path.splitext(os.path.basename(rgb_overlay_path))[0]
    else:
        rgb_overlay_path_str = "None"
    r, p, y = quat2euler(robot_init_quat)
    video_path = f"{scene_name}/{control_mode}/{env_save_name}/rob_{robot_init_x}_{robot_init_y}_rot_{r:.3f}_{p:.3f}_{y:.3f}_rgb_overlay_{rgb_overlay_path_str}/{video_name}"
    video_path = os.path.join(logging_dir, video_path)
    write_video(video_path, images, fps=5)

    # save action trajectory
    action_path = video_path.replace(".mp4", ".png")
    action_root = os.path.dirname(action_path) + "/actions/"
    os.makedirs(action_root, exist_ok=True)
    action_path = action_root + os.path.basename(action_path)
    model.visualize_epoch(predicted_actions, images, save_path=action_path)
    return success == "success"


def maniskill2_evaluator(model, args):
    control_mode = get_robot_control_mode(args.robot, args.policy_model)
    success_arr = []

    save_rollout_dir = getattr(args, "save_rollout_dir", None)

    # run inference
    for robot_init_x in args.robot_init_xs:
        for robot_init_y in args.robot_init_ys:
            for robot_init_quat in args.robot_init_quats:
                kwargs = dict(
                    model=model,
                    ckpt_path=args.ckpt_path,
                    robot_name=args.robot,
                    env_name=args.env_name,
                    scene_name=args.scene_name,
                    robot_init_x=robot_init_x,
                    robot_init_y=robot_init_y,
                    robot_init_quat=robot_init_quat,
                    control_mode=control_mode,
                    additional_env_build_kwargs=args.additional_env_build_kwargs,
                    rgb_overlay_path=args.rgb_overlay_path,
                    control_freq=args.control_freq,
                    sim_freq=args.sim_freq,
                    max_episode_steps=args.max_episode_steps,
                    enable_raytracing=args.enable_raytracing,
                    additional_env_save_tags=args.additional_env_save_tags,
                    obs_camera_name=args.obs_camera_name,
                    logging_dir=args.logging_dir,
                    save_rollout_dir=save_rollout_dir,
                )
                if args.obj_variation_mode == "xy":
                    for obj_init_x in args.obj_init_xs:
                        for obj_init_y in args.obj_init_ys:
                            success_arr.append(
                                run_maniskill2_eval_single_episode(
                                    obj_init_x=obj_init_x,
                                    obj_init_y=obj_init_y,
                                    **kwargs,
                                )
                            )
                elif args.obj_variation_mode == "episode":
                    for obj_episode_id in range(
                        args.obj_episode_range[0], args.obj_episode_range[1]
                    ):
                        success_arr.append(
                            run_maniskill2_eval_single_episode(
                                obj_episode_id=obj_episode_id, **kwargs
                            )
                        )
                else:
                    raise NotImplementedError()

    return success_arr
