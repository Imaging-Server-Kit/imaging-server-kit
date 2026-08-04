def napari_available():
    """Check if the Napari dependencies are installed."""
    try:
        import imaging_server_kit.gui.napari_serverkit
        return True
    except ImportError as e:
        # raise ImportError(
        #     "This feature requires the Imaging Server Kit `napari` optional dependencies.\n"
        #     "Install them with:\n"
        #     "    pip install imaging-server-kit[napari]"
        # ) from e
        return False

def qupath_available():
    """Check if the QuPath dependencies (e.g. QuBaLab) are installed."""
    try:
        import imaging_server_kit.gui.qupath_serverkit
        return True
    except ImportError as e:
        # raise ImportError(
        #     "This feature requires the Imaging Server Kit `qupath` optional dependencies.\n"
        #     "Install them with:\n"
        #     "    pip install imaging-server-kit[qupath]"
        # ) from e
        return False

def remote_available():
    """Check if the remote dependencies (e.g. FastAPI) are installed."""
    try:
        import imaging_server_kit.remote
        return True
    except ImportError as e:
        # raise ImportError(
        #     "This feature requires the Imaging Server Kit `remote` optional dependencies.\n"
        #     "Install them with:\n"
        #     "    pip install imaging-server-kit[remote]"
        # ) from e
        return False