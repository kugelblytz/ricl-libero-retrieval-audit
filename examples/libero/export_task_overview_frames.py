import dataclasses
import math
import pathlib
import typing

from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import tyro


LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]


@dataclasses.dataclass
class Args:
    task_suite_name: str = "libero_90"
    task_ids: typing.Optional[str] = "0-39"
    output_dir: str = "data/libero/task_overviews/libero90_0_39"
    resolution: int = 256
    num_steps_wait: int = 10
    seed: int = 7
    rotate_180: bool = False
    contact_sheet_cols: int = 5
    train_task_ids: typing.Optional[str] = "0-19"
    heldout_task_ids: typing.Optional[str] = "20-39"


def main(args: Args) -> None:
    output_dir = pathlib.Path(args.output_dir)
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    task_ids = _parse_task_ids(args.task_ids, task_suite.n_tasks)
    train_task_ids = set(_parse_task_ids(args.train_task_ids, task_suite.n_tasks)) if args.train_task_ids else set()
    heldout_task_ids = (
        set(_parse_task_ids(args.heldout_task_ids, task_suite.n_tasks)) if args.heldout_task_ids else set()
    )

    rows = []
    frame_paths = []
    for task_id in task_ids:
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = _get_libero_env(task, args.resolution, args.seed)
        try:
            env.reset()
            obs = env.set_init_state(initial_states[0])
            for _ in range(args.num_steps_wait):
                obs, _reward, _done, _info = env.step(LIBERO_DUMMY_ACTION)

            image = np.ascontiguousarray(obs["agentview_image"])
            if args.rotate_180:
                image = image[::-1, ::-1]
            frame_path = frames_dir / f"task_{task_id:02d}.png"
            Image.fromarray(image).save(frame_path)
            split = _split_for_task(task_id, train_task_ids, heldout_task_ids)
            frame_paths.append((task_id, split, task_description, frame_path))
            rows.append((task_id, split, task_description, frame_path.relative_to(output_dir)))
        finally:
            env.close()

    _write_index(output_dir, rows)
    _write_contact_sheet(output_dir / "contact_sheet.png", frame_paths, cols=args.contact_sheet_cols)
    print(f"Wrote frames to {frames_dir}")
    print(f"Wrote index to {output_dir / 'index.md'}")
    print(f"Wrote contact sheet to {output_dir / 'contact_sheet.png'}")


def _get_libero_env(task, resolution: int, seed: int):
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, task_description


def _parse_task_ids(task_ids: typing.Optional[str], num_tasks_in_suite: int) -> typing.List[int]:
    if task_ids is None:
        return list(range(num_tasks_in_suite))

    parsed = []
    for part in task_ids.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_raw, end_raw = part.split("-", maxsplit=1)
            parsed.extend(range(int(start_raw), int(end_raw) + 1))
        else:
            parsed.append(int(part))

    invalid = [task_id for task_id in parsed if task_id < 0 or task_id >= num_tasks_in_suite]
    if invalid:
        raise ValueError(f"task_ids out of range for suite with {num_tasks_in_suite} tasks: {invalid}")
    return parsed


def _split_for_task(task_id: int, train_task_ids: set, heldout_task_ids: set) -> str:
    if task_id in train_task_ids:
        return "train"
    if task_id in heldout_task_ids:
        return "held-out"
    return "other"


def _write_index(
    output_dir: pathlib.Path,
    rows: typing.List[typing.Tuple[int, str, str, pathlib.Path]],
) -> None:
    lines = ["# LIBERO Task Overview Frames", "", "| Task | Split | Instruction | Frame |", "|---:|---|---|---|"]
    for task_id, split, description, frame_path in rows:
        lines.append(f"| {task_id} | {split} | {description} | [{frame_path}]({frame_path}) |")
    (output_dir / "index.md").write_text("\n".join(lines) + "\n")


def _write_contact_sheet(
    output_path: pathlib.Path,
    frame_paths: typing.List[typing.Tuple[int, str, str, pathlib.Path]],
    *,
    cols: int,
) -> None:
    thumb_size = 256
    label_height = 104
    rows = math.ceil(len(frame_paths) / cols)
    sheet = Image.new("RGB", (cols * thumb_size, rows * (thumb_size + label_height)), "white")

    for idx, (task_id, split, description, frame_path) in enumerate(frame_paths):
        row = idx // cols
        col = idx % cols
        frame = Image.open(frame_path).convert("RGB").resize((thumb_size, thumb_size))
        x = col * thumb_size
        y = row * (thumb_size + label_height)
        sheet.paste(frame, (x, y + label_height))
        _draw_label(sheet, x, y, thumb_size, label_height, task_id, split, description)

    sheet.save(output_path)


def _draw_label(
    sheet: Image.Image,
    x: int,
    y: int,
    width: int,
    height: int,
    task_id: int,
    split: str,
    description: str,
) -> None:
    draw = ImageDraw.Draw(sheet)
    split_colors = {
        "train": (224, 242, 254),
        "held-out": (240, 253, 244),
        "other": (245, 245, 245),
    }
    fill = split_colors.get(split, split_colors["other"])
    draw.rectangle((x, y, x + width, y + height), fill=fill)
    draw.rectangle((x, y, x + width - 1, y + height - 1), outline=(180, 180, 180))

    font = ImageFont.load_default()
    draw.text((x + 8, y + 8), f"task_{task_id:02d} | {split}", fill=(15, 23, 42), font=font)
    for line_idx, line in enumerate(_wrap_text(description, max_chars=34)[:4]):
        draw.text((x + 8, y + 28 + line_idx * 16), line, fill=(20, 20, 20), font=font)


def _wrap_text(text: str, max_chars: int) -> typing.List[str]:
    words = text.split()
    lines = []
    current = ""
    for word in words:
        candidate = word if not current else f"{current} {word}"
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


if __name__ == "__main__":
    tyro.cli(main)
