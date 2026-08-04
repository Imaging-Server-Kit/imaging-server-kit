import argparse


def cmd_demo_napari(args):
    from imaging_server_kit.core.check_install import napari_available

    if not napari_available():
        print(
            "To use this method, install the Imaging Server Kit package with `pip install imaging-server-kit[all]`."
        )
        return

    from imaging_server_kit.gui import to_napari
    from imaging_server_kit.demo import multi_algo_demos
    import napari

    to_napari(multi_algo_demos)
    napari.run()


def cmd_demo_serve(args):
    from imaging_server_kit.core.check_install import remote_available
    
    if not remote_available():
        print(
            "To use this method, install the Imaging Server Kit package with `pip install imaging-server-kit[all]`."
        )
        return
    
    from imaging_server_kit.demo import multi_algo_demos
    from imaging_server_kit.remote import serve

    serve(multi_algo_demos)


def cmd_tools_napari(args):
    from imaging_server_kit.core.check_install import napari_available

    if not napari_available():
        print(
            "To use this method, install the Imaging Server Kit package with `pip install imaging-server-kit[all]`."
        )
        return

    from imaging_server_kit.demo import multi_algo_tools
    from imaging_server_kit.gui.napari_serverkit import to_napari
    import napari

    to_napari(multi_algo_tools)
    napari.run()


def cmd_tools_serve(args):
    from imaging_server_kit.core.check_install import remote_available
        
    if not remote_available():
        print(
            "To use this method, install the Imaging Server Kit package with `pip install imaging-server-kit[all]`."
        )
        return
        
    from imaging_server_kit.demo import multi_algo_tools
    from imaging_server_kit.remote import serve

    serve(multi_algo_tools)


def cmd_qupath(args):
    from imaging_server_kit.core.check_install import qupath_available
    
    if not qupath_available():
        print(
            "To use this method, install the Imaging Server Kit package with `pip install imaging-server-kit[all]`."
        )
        return
    
    from imaging_server_kit.remote import Client
    from imaging_server_kit.gui.qupath_serverkit import to_qupath

    to_qupath(runner=Client())


def main(argv=None):
    parser = argparse.ArgumentParser(description="Imaging Server Kit CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # serverkit demo <subcommand>
    p_demo = subparsers.add_parser("demo", help="Run demo algorithms.")
    demo_sub = p_demo.add_subparsers(dest="demo_command", required=True)

    # serverkit demo napari
    p_demo_napari = demo_sub.add_parser("napari", help="Start the Napari demo")
    p_demo_napari.set_defaults(func=cmd_demo_napari)

    # serverkit demo serve
    p_demo_serve = demo_sub.add_parser("serve", help="Start the server demo")
    p_demo_serve.set_defaults(func=cmd_demo_serve)

    # serverkit tools <subcommand>
    p_tools = subparsers.add_parser("tools", help="Run tool algorithms.")
    tools_sub = p_tools.add_subparsers(dest="tools_command", required=True)

    # serverkit tools napari
    p_tools_napari = tools_sub.add_parser("napari", help="Start the Napari tools")
    p_tools_napari.set_defaults(func=cmd_tools_napari)

    # serverkit tools serve
    p_tools_serve = tools_sub.add_parser("serve", help="Start the server tools")
    p_tools_serve.set_defaults(func=cmd_tools_serve)
    
    # serverkit qupath
    qp_parser = subparsers.add_parser("qupath", help="Connect to QuPath.")
    qp_parser.set_defaults(func=cmd_qupath)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    main()
