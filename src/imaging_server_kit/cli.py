import argparse


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
    

def cmd_tools_napari(args):
    from imaging_server_kit.core.errors import napari_available
    
    if not napari_available():
        print(
            "To use this method, install the Imaging Server Kit Napari plugin with `pip install napari-serverkit`."
        )
        return
    
    import imaging_server_kit as sk
    from imaging_server_kit.demo import multi_algo_tools
    import napari

    sk.to_napari(multi_algo_tools)
    napari.run()


def cmd_tools_serve(args):
    import imaging_server_kit as sk
    from imaging_server_kit.demo import multi_algo_tools

    sk.serve(multi_algo_tools)


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

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    main()
