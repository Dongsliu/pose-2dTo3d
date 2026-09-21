import argparse
import json
import sys
from pathlib import Path

import bpy
from mathutils import Vector


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--pose", required=True)
    parser.add_argument("--template", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args(argv)


def midpoint(a, b):
    return (a + b) * 0.5


def extend_from(a, b, length_ratio=0.18):
    direction = a - b
    if direction.length < 1e-6:
        return a + Vector((0.0, 0.0, 0.01))
    return a + direction.normalized() * max(direction.length * length_ratio, 0.01)


def load_targets(pose_path):
    with open(pose_path, "r", encoding="utf-8") as handle:
        pose = json.load(handle)

    points_3d = pose.get("keypoints_3d") or []
    points_2d = pose.get("keypoints_2d") or []
    if len(points_3d) < 17 or len(points_2d) < 17:
        raise ValueError("pose.json must contain at least 17 keypoints in keypoints_2d and keypoints_3d")

    raw_3d = [Vector((float(p[0]), float(p[1]), float(p[2]))) for p in points_3d]
    raw_2d = [Vector((float(p[0]), float(p[1]), 0.0)) for p in points_2d]

    hip_3d = midpoint(raw_3d[11], raw_3d[12])
    chest_3d = midpoint(raw_3d[5], raw_3d[6])
    body_length = max((chest_3d - hip_3d).length, 1.0)
    scale = 2.0 / body_length

    def convert(index):
        point = raw_3d[index] - hip_3d
        return Vector(
            (
                point.x * scale,
                point.z * scale,
                -point.y * scale,
            )
        )

    return [convert(index) for index in range(len(raw_3d))]


def make_targets(points):
    def center(a, b):
        return midpoint(points[a], points[b])

    joints = {
        "hip": center(11, 12),
        "chest": center(5, 6),
        "neck": points[0],
        "head": points[0],
        "shoulder.L": points[5],
        "shoulder.R": points[6],
        "elbow.L": points[7],
        "elbow.R": points[8],
        "wrist.L": points[9],
        "wrist.R": points[10],
        "hip.L": points[11],
        "hip.R": points[12],
        "knee.L": points[13],
        "knee.R": points[14],
        "ankle.L": points[15],
        "ankle.R": points[16],
    }
    targets = {}

    def add(name, head, tail):
        targets[name] = (head, tail)

    spine_points = [
        joints["hip"],
        joints["hip"].lerp(joints["chest"], 0.25),
        joints["hip"].lerp(joints["chest"], 0.5),
        joints["hip"].lerp(joints["chest"], 0.75),
        joints["chest"],
        joints["chest"].lerp(joints["neck"], 0.5),
        joints["neck"],
        joints["head"] + Vector((0.0, 0.0, 0.12)),
    ]
    for index in range(7):
        name = "spine" if index == 0 else f"spine.{index:03d}"
        add(name, spine_points[index], spine_points[index + 1])
    add("face", joints["neck"], joints["head"] + Vector((0.0, 0.0, 0.12)))

    for side in ("L", "R"):
        add(f"shoulder.{side}", joints["chest"], joints[f"shoulder.{side}"])
        add(f"upper_arm.{side}", joints[f"shoulder.{side}"], joints[f"elbow.{side}"])
        add(f"forearm.{side}", joints[f"elbow.{side}"], joints[f"wrist.{side}"])
        add(
            f"hand.{side}",
            joints[f"wrist.{side}"],
            extend_from(joints[f"wrist.{side}"], joints[f"elbow.{side}"]),
        )
        add(f"pelvis.{side}", joints["hip"], joints[f"hip.{side}"])
        add(f"thigh.{side}", joints[f"hip.{side}"], joints[f"knee.{side}"])
        add(f"shin.{side}", joints[f"knee.{side}"], joints[f"ankle.{side}"])

    if len(points) >= 23:
        foot_l = midpoint(points[17], points[18])
        foot_r = midpoint(points[20], points[21])
        add("foot.L", joints["ankle.L"], foot_l)
        add("toe.L", foot_l, points[17])
        add("heel.02.L", joints["ankle.L"], points[19])
        add("foot.R", joints["ankle.R"], foot_r)
        add("toe.R", foot_r, points[20])
        add("heel.02.R", joints["ankle.R"], points[22])

    def add_hand(side, base):
        wrist = joints[f"wrist.{side}"]
        palm_names = ("palm.01", "palm.02", "palm.03", "palm.04")
        palm_indices = (base + 5, base + 9, base + 13, base + 17)
        for name, index in zip(palm_names, palm_indices):
            add(f"{name}.{side}", wrist, points[index])

        fingers = {
            "thumb": (base + 1, base + 2, base + 3, base + 4),
            "f_index": (base + 5, base + 6, base + 7, base + 8),
            "f_middle": (base + 9, base + 10, base + 11, base + 12),
            "f_ring": (base + 13, base + 14, base + 15, base + 16),
            "f_pinky": (base + 17, base + 18, base + 19, base + 20),
        }
        for name, indices in fingers.items():
            for segment in range(3):
                add(
                    f"{name}.{segment + 1:02d}.{side}",
                    points[indices[segment]],
                    points[indices[segment + 1]],
                )

    if len(points) >= 133:
        add_hand("L", 91)
        add_hand("R", 112)

    return targets


def find_armature():
    armatures = [obj for obj in bpy.data.objects if obj.type == "ARMATURE"]
    for obj in armatures:
        if obj.name == "PoseRig":
            return obj
    if not armatures:
        raise RuntimeError("No armature found in PoseRig.blend")
    return armatures[0]


def apply_pose(armature, targets):
    bpy.ops.object.mode_set(mode="OBJECT") if armature.mode != "OBJECT" else None
    bpy.ops.object.select_all(action="DESELECT")
    armature.select_set(True)
    bpy.context.view_layer.objects.active = armature

    # This export is a static pose.  Writing the detected joints directly to
    # Edit Bones keeps the template names and hierarchy while avoiding the
    # incompatible Rest Pose offsets of the metarig.
    armature.animation_data_clear()
    bpy.ops.object.mode_set(mode="EDIT")
    applied = 0
    for name, (head, tail) in targets.items():
        edit_bone = armature.data.edit_bones.get(name)
        direction = tail - head
        if edit_bone is None or direction.length < 1e-6:
            continue
        edit_bone.head = head
        edit_bone.tail = tail
        edit_bone.use_connect = False
        applied += 1

    bpy.ops.object.mode_set(mode="OBJECT")
    return applied


def export_fbx(armature, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.object.select_all(action="DESELECT")
    armature.select_set(True)
    bpy.context.view_layer.objects.active = armature
    bpy.ops.export_scene.fbx(
        filepath=str(output_path),
        use_selection=True,
        object_types={"ARMATURE"},
        add_leaf_bones=False,
        bake_anim=False,
        bake_anim_use_all_bones=True,
        bake_anim_use_nla_strips=False,
        bake_anim_use_all_actions=False,
        apply_scale_options="FBX_SCALE_ALL",
    )


def main():
    args = parse_args()
    bpy.ops.wm.open_mainfile(filepath=str(Path(args.template).resolve()))
    points = load_targets(args.pose)
    targets = make_targets(points)
    armature = find_armature()
    applied = apply_pose(armature, targets)
    export_fbx(armature, Path(args.output).resolve())
    print(f"POSE_FBX_EXPORTED {applied}")


if __name__ == "__main__":
    main()
