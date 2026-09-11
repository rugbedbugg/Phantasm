"""Install and test both distribution formats in clean environments outside checkout."""

import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path


def run(*command: str, cwd: Path, env: dict) -> None:
    subprocess.run(command, cwd=cwd, env=env, check=True)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    artifacts = [*root.joinpath("dist").glob("*.tar.gz"), *root.joinpath("dist").glob("*.whl")]
    if len(artifacts) != 2:
        raise SystemExit("Build exactly one sdist and one wheel in dist/ first")
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.pop("VIRTUAL_ENV", None)
    with tempfile.TemporaryDirectory(prefix="phantasm-dist-") as directory:
        work = Path(directory)
        deps = work / "dependencies.txt"
        run(
            "uv",
            "export",
            "--quiet",
            "--locked",
            "--extra",
            "dev",
            "--no-emit-project",
            "--format",
            "requirements-txt",
            "--output-file",
            str(deps),
            cwd=root,
            env=env,
        )
        sdist = next(path for path in artifacts if path.name.endswith(".tar.gz"))
        with tarfile.open(sdist) as archive:
            for member in archive.getmembers():
                destination = (work / "source" / member.name).resolve()
                if (
                    not destination.is_relative_to(work / "source")
                    or member.issym()
                    or member.islnk()
                ):
                    raise ValueError("Unsafe source distribution member")
            archive.extractall(work / "source")
        source = next((work / "source").iterdir())
        for name in (
            "scripts/train.py",
            "notebooks/finetune.ipynb",
            "src/phantasm/resources/training.txt",
            "docs/TRAINING.md",
            "docs/COLAB.md",
            "examples/demo.json",
        ):
            if not (source / name).is_file():
                raise ValueError(f"Source distribution is missing {name}")
        for index, artifact in enumerate(artifacts):
            directory = work / str(index)
            directory.mkdir()
            venv = directory / "venv"
            run("uv", "venv", str(venv), "--python", sys.executable, cwd=directory, env=env)
            python = str(venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python"))
            run(
                "uv",
                "pip",
                "sync",
                "--python",
                python,
                "--require-hashes",
                str(deps),
                cwd=directory,
                env=env,
            )
            run(
                "uv",
                "pip",
                "install",
                "--python",
                python,
                "--no-deps",
                str(artifact),
                cwd=directory,
                env=env,
            )
            shutil.copytree(source / "tests", directory / "tests")
            for wrapper in (
                "scrape_discord.py",
                "parse_discord_export.py",
                "format_training_data.py",
                "inference.py",
            ):
                shutil.copy(source / wrapper, directory / wrapper)
                run(python, wrapper, "--help", cwd=directory, env=env)
            run(python, str(source / "scripts/train.py"), "--help", cwd=directory, env=env)
            run(
                python,
                "-c",
                "import pathlib, sys, phantasm; "
                "package = pathlib.Path(phantasm.__file__).parent; "
                "assert package.is_relative_to(sys.prefix); "
                "assert (package / 'resources/training.txt').is_file()",
                cwd=directory,
                env=env,
            )
            run(python, "-m", "phantasm.cli", "--help", cwd=directory, env=env)
            for command in (
                "scrape",
                "parse",
                "format",
                "inspect",
                "audit",
                "evaluate",
                "chat",
                "train",
                "colab",
                "recover",
            ):
                run(python, "-m", "phantasm.cli", command, "--help", cwd=directory, env=env)
            run(python, "-m", "pytest", "-q", "tests", cwd=directory, env=env)
            print(f"Validated installed artifact: {artifact.name}")


if __name__ == "__main__":
    main()
