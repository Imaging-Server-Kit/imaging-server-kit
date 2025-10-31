from pathlib import Path
import argparse


def cmd_new(args):
    from cookiecutter.main import cookiecutter

    output_dir = str(Path(args.output_dir).resolve())
    template_dir = str(Path(__file__).parent.resolve() / "template")
    cookiecutter(template=template_dir, output_dir=output_dir)


def cmd_demo_napari(args):
    from imaging_server_kit.core.errors import napari_available
    
    if not napari_available():
        print(
            "To use this method, install the Imaging Server Kit Napari plugin with `pip install napari-serverkit`."
        )
        return
    
    import imaging_server_kit as sk
    from imaging_server_kit.demo import multi_algo_examples
    import napari

    sk.to_napari(multi_algo_examples)
    napari.run()


def cmd_demo_serve(args):
    import imaging_server_kit as sk
    from imaging_server_kit.demo import multi_algo_examples

    sk.serve(multi_algo_examples)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Imaging Server Kit CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # serverkit new <output_dir>
    p_new = subparsers.add_parser(
        "new", help="Generate a new project structure for an algorithm project."
    )
    p_new.add_argument("output_dir", help="Output directory.")
    p_new.set_defaults(func=cmd_new)

    # serverkit demo <subcommand>
    p_demo = subparsers.add_parser("demo", help="Run demo algorithms.")
    demo_sub = p_demo.add_subparsers(dest="demo_command", required=True)

    # serverkit demo napari
    p_demo_napari = demo_sub.add_parser("napari", help="Start the Napari demo")
    p_demo_napari.set_defaults(func=cmd_demo_napari)

    # serverkit demo serve
    p_demo_serve = demo_sub.add_parser("serve", help="Start the server demo")
    p_demo_serve.set_defaults(func=cmd_demo_serve)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    main()
