from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import sapien

from ._base_task import Base_Task
from .utils import *


class replay_block_auto_init(Base_Task):
    """Replay task that aligns the target block and active arm to first-frame metadata."""

    def setup_demo(self, **kwags):
        self._task_kwargs = dict(kwags)
        self.replay_finished = False
        self.replay_failed = False
        self.replay_init_meta = None
        self.target_object = None
        self.static_scene_objects = []
        super()._init_task_env_(**kwags)
        self._run_auto_init()

    def load_actors(self):
        spawn_pose = self._task_kwargs.get(
            "target_object_spawn_pose",
            [0.0, 0.20, 0.82, 1.0, 0.0, 0.0, 0.0],
        )
        modelname = self._task_kwargs.get("target_object_modelname", "replace_with_robotwin_modelname")
        model_id = int(self._task_kwargs.get("target_object_model_id", 0))
        convex = bool(self._task_kwargs.get("target_object_convex", True))
        self.target_object = create_actor(
            self,
            pose=self._as_sapien_pose(spawn_pose),
            modelname=modelname,
            convex=convex,
            model_id=model_id,
        )

        for obj_cfg in self._task_kwargs.get("static_scene_objects", []) or []:
            actor = create_actor(
                self,
                pose=self._as_sapien_pose(obj_cfg["pose"]),
                modelname=obj_cfg["modelname"],
                convex=bool(obj_cfg.get("convex", True)),
                model_id=int(obj_cfg.get("model_id", 0)),
                is_static=bool(obj_cfg.get("is_static", True)),
            )
            self.static_scene_objects.append(actor)

        if self.target_object is not None:
            self.add_prohibit_area(self.target_object, padding=0.05)

    def play_once(self):
        self.info["info"] = {
            "{A}": getattr(self.target_object, "name", "target_object"),
        }
        return self.info

    def check_success(self):
        return bool(getattr(self, "replay_finished", False))

    def _run_auto_init(self):
        meta = self._load_init_meta()
        self.replay_init_meta = meta
        self._apply_target_object_pose(meta)
        self._apply_eef_pose(meta)
        self.robot.set_origin_endpose()

        if hasattr(self.robot, "left_entity"):
            self.replay_last_left_qpos = np.array(self.robot.left_entity.get_qpos(), dtype=np.float64)
        if hasattr(self.robot, "right_entity"):
            self.replay_last_right_qpos = np.array(self.robot.right_entity.get_qpos(), dtype=np.float64)

        self.get_obs()
        print(
            f"[replay_block_auto_init] scene aligned from {meta.get('first_frame_image')} "
            f"arm={meta.get('replay_arm')}",
            flush=True,
        )

    def _load_init_meta(self) -> dict:
        meta_path = self._task_kwargs.get("init_meta_path")
        if self._task_kwargs.get("init_meta_from_env", True):
            env_var = self._task_kwargs.get("init_meta_env_var", "REPLAY_INIT_META_PATH")
            meta_path = os.environ.get(env_var, meta_path)
        if not meta_path:
            raise RuntimeError(
                "Auto-init metadata path is missing. Set REPLAY_INIT_META_PATH "
                "or task_config.init_meta_path before launching Replay_Policy."
            )

        meta_path = Path(meta_path)
        if not meta_path.is_file():
            raise FileNotFoundError(f"Auto-init metadata file not found: {meta_path}")
        return json.loads(meta_path.read_text(encoding="utf-8"))

    def _apply_target_object_pose(self, meta: dict):
        pose7d = meta.get("robotwin_obj_pose7d_wxyz")
        if pose7d is None:
            raise KeyError("robotwin_obj_pose7d_wxyz is missing from init_meta")
        self._set_actor_pose_generic(self.target_object, self._as_sapien_pose(pose7d))
        self.delay(1)

    def _apply_eef_pose(self, meta: dict):
        arm_tag = ArmTag(meta.get("replay_arm", self._task_kwargs.get("replay_arm", "right")))
        pose7d = meta.get("robotwin_eef_pose7d_wxyz")
        if pose7d is None:
            raise KeyError("robotwin_eef_pose7d_wxyz is missing from init_meta")
        self.move(self.move_to_pose(arm_tag, pose7d))
        self.get_obs()

    def _resolve_actor_backend(self, actor):
        if actor is None:
            return None
        if hasattr(actor, "set_pose"):
            return actor
        for attr_name in ("entity", "actor", "instance", "_actor", "_entity", "body", "object"):
            backend = getattr(actor, attr_name, None)
            if backend is not None and hasattr(backend, "set_pose"):
                return backend
        return actor

    def _set_actor_pose_generic(self, actor, pose):
        backend = self._resolve_actor_backend(actor)
        if backend is None or not hasattr(backend, "set_pose"):
            raise AttributeError(f"Cannot locate set_pose backend for actor: {actor}")
        backend.set_pose(pose)
        if hasattr(backend, "set_velocity"):
            backend.set_velocity(np.zeros(3))
        if hasattr(backend, "set_angular_velocity"):
            backend.set_angular_velocity(np.zeros(3))

    @staticmethod
    def _as_sapien_pose(values):
        if isinstance(values, sapien.Pose):
            return values
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        if values.size != 7:
            raise ValueError(f"Expected 7D pose [x, y, z, qw, qx, qy, qz], got {values}")
        return sapien.Pose(values[:3], values[3:7])
