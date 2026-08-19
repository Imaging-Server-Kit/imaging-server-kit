import argparse


def cmd_demo_napari(args):
    from imaging_server_kit.gui import to_napari
    from imaging_server_kit.demo import multi_algo_demos
    import napari

    to_napari(multi_algo_demos)
    napari.run()


def cmd_demo_serve(args):
    from imaging_server_kit.demo import multi_algo_demos
    from imaging_server_kit.remote import serve

    serve(multi_algo_demos)


def cmd_tools_napari(args):
    from imaging_server_kit.demo import multi_algo_tools
    from imaging_server_kit.gui.napari_serverkit import to_napari
    import napari

    to_napari(multi_algo_tools)
    napari.run()


def cmd_tools_serve(args):
    from imaging_server_kit.demo import multi_algo_tools
    from imaging_server_kit.remote import serve

    serve(multi_algo_tools)


def cmd_qupath(with_napari: bool = False):
    from imaging_server_kit.remote import Client

    try:
        from imaging_server_kit.gui.qupath_serverkit import to_qupath
    except ImportError as e:
        raise ImportError(
            "This feature requires the Imaging Server Kit `qupath` optional dependencies.\n"
            "Install them with:\n"
            "    pip install imaging-server-kit[qupath]"
        ) from e

    if with_napari:
        import napari
        viewer = napari.Viewer()
    else:
        viewer = None
    
    to_qupath(runner=Client(), viewer=viewer)
    
    if with_napari:
        napari.run()


def main(
        # argv=None
    ):
    parser = argparse.ArgumentParser(description="Imaging Server Kit CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # serverkit demo <subcommand>
    p_demo = subparsers.add_parser("demo", help="Run demo algorithms")
    demo_sub = p_demo.add_subparsers(dest="demo_command", required=True)

    # serverkit demo napari
    p_demo_napari = demo_sub.add_parser("napari", help="Start the Napari demo")
    p_demo_napari.set_defaults(func=cmd_demo_napari)

    # serverkit demo serve
    p_demo_serve = demo_sub.add_parser("serve", help="Start the server demo")
    p_demo_serve.set_defaults(func=cmd_demo_serve)

    # serverkit tools <subcommand>
    p_tools = subparsers.add_parser("tools", help="Run tool algorithms")
    tools_sub = p_tools.add_subparsers(dest="tools_command", required=True)

    # serverkit tools napari
    p_tools_napari = tools_sub.add_parser("napari", help="Start the Napari tools")
    p_tools_napari.set_defaults(func=cmd_tools_napari)

    # serverkit tools serve
    p_tools_serve = tools_sub.add_parser("serve", help="Start the server tools")
    p_tools_serve.set_defaults(func=cmd_tools_serve)

    # serverkit qupath
    qp_parser = subparsers.add_parser("qupath", help="Connect to QuPath")
    qp_parser.add_argument(
        "--with-napari",
        action="store_true",
        help="Use this flag to integrate a Napari viewer and see results in it along with the QuPath GUI.",
    )

    args = parser.parse_args()
    
    if args.command == "qupath":
        cmd_qupath(with_napari=args.with_napari)
    else:
        return args.func(args)


if __name__ == "__main__":
    main()
