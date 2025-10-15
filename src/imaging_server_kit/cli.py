from pathlib import Path
import argparse


def main():
    parser = argparse.ArgumentParser(description="Imaging Server Kit CLI")
    subparsers = parser.add_subparsers(dest="command")

    new_algo_parser = subparsers.add_parser(
        "new",
        help="Generate a new project structure for an algorithm server.",
    )
    new_algo_parser.add_argument(
        "output_dir",
        help="Output directory.",
    )

    subparsers.add_parser(
        "demo",
        help="Run a demo algorithm server.",
    )

    args = parser.parse_args()

    if args.command == "new":
        from cookiecutter.main import cookiecutter
        output_dir = Path(args.output_dir).resolve()
        template_dir = str(Path(__file__).parent.resolve() / "template")
        cookiecutter(template=template_dir, output_dir=output_dir)
    elif args.command == "demo":
        import imaging_server_kit as sk
        from imaging_server_kit.demo import threshold
        sk.serve(threshold, host="0.0.0.0", port=8000)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
