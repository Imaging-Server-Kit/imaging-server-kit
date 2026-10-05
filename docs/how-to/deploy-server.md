# Serve an algorithm

An algorithm server can run on a workstation with a GPU, a cluster node, or a small single-board computer, for example. Clients then connect to it over the network from Napari, QuPath, or Python.

!!! warning "Security"
    Only connect to servers you trust, or run servers on networks you trust. Algorithm servers have no authentication, and no limit on the size of the responses that clients accept.

## Choosing the host and port

`sk.serve()` accepts `host` and `port` arguments:

```python
if __name__ == "__main__":
    sk.serve(my_algo, host="0.0.0.0", port=8000)
```

- `host` defaults to `"0.0.0.0"`: the server listens on all network interfaces, so it is reachable from other machines on the network. Use `host="127.0.0.1"` to only accept connections from the same machine.
- `port` defaults to `8000`. If the port is already in use, the server prints a message and doesn't start; choose another port, for example `port=8001`.

## Connecting from another machine

Clients connect with the address of the server machine instead of `localhost`, for example `http://192.168.1.42:8000`:

- In Napari, enter the address in the *Server URL* field of `Plugins > Imaging Server Kit > Connect to server`.
- In Python, you can pass it to `sk.Client("http://192.168.1.42:8000")`.

**Things to check**

- The port should be open in the firewall of the server machine.
- The client and the server should use the same version of `imaging-server-kit`. You can check the server version at the `/version` endpoint - for example, http://localhost:8000/version.

See [HTTP endpoints](../reference/http.md) for the other routes exposed by the server.
