import argparse
import sys
from pathlib import Path

from .config import Settings
from .postprocess import process
from .presets import load_presets
from .prompting import to_english


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="falak3d")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("generate", help="génère un objet et l'écrit en .glb")
    p.add_argument("prompt")
    p.add_argument("--preset", default="ecommerce")
    p.add_argument("-o", "--out", type=Path, default=Path("out/objet.glb"))
    p.add_argument("--fake", action="store_true", help="primitive au lieu de Shap-E (sans GPU)")

    p = sub.add_parser("serve", help="lance l'API")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8001)

    sub.add_parser("presets", help="liste les presets")

    args = parser.parse_args(argv)
    settings = Settings.from_env()

    if args.cmd == "presets":
        for preset in load_presets(settings.presets_dir).values():
            print(f"{preset.name:<12} {preset.title} — {preset.description}")
        return 0

    if args.cmd == "serve":
        import uvicorn

        from .api import create_app
        uvicorn.run(create_app(settings), host=args.host, port=args.port)
        return 0

    from .generator import FakeGenerator, ShapEGenerator

    preset = load_presets(settings.presets_dir)[args.preset]
    caption = preset.prompt(to_english(args.prompt, settings.ollama_url, settings.ollama_model))
    print(f"description : {caption}")
    generator = FakeGenerator() if args.fake else ShapEGenerator(settings.data_dir / "shap_e_cache")
    mesh, report = process(generator.generate(caption, preset), preset.target_faces,
                           preset.size_m)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(args.out, file_type="glb")
    print(f"{args.out}  {report.faces_in} -> {report.faces_out} faces, "
          f"{report.size_m[0]} x {report.size_m[1]} x {report.size_m[2]} m")
    return 0


if __name__ == "__main__":
    sys.exit(main())
