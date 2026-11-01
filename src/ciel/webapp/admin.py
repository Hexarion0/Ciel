"""Authenticated browser administration, shared with the active hosted assistant."""

import threading


class WebAdministration:
    """Handle explicit owner UI actions through already authenticated HTTP handlers."""

    def __init__(self, ui, settings, network, *, restart, memories, forget) -> None:
        self.ui, self.settings, self.network = ui, settings, network
        self.restart, self.memories, self.forget = restart, memories, forget

    def get(self, handler) -> bool:
        if handler.path == "/api/settings":
            handler._json(200, self.settings.snapshot())
        elif handler.path == "/api/network":
            handler._json(200, self.network.snapshot())
        elif handler.path == "/api/memory":
            handler._json(200, {"memories": self.memories()})
        elif handler.path == "/api/network/certificate":
            try:
                handler.response_headers = {
                    "Content-Disposition": 'attachment; filename="ciel-ca.crt"'
                }
                handler._send(200, self.network.certificate(), "application/x-x509-ca-cert")
            except OSError:
                handler._json(404, {"error": "Enable the local phone connection first."})
        else:
            return False
        return True

    def post(self, handler) -> bool:
        paths = {"/api/settings", "/api/network/setup", "/api/server/restart", "/api/memory/forget"}
        if handler.path not in paths:
            return False
        try:
            data = self.ui.channels.json(handler)
            if handler.path == "/api/settings":
                handler._json(200, self.settings.save(data))
            elif handler.path == "/api/network/setup":
                if handler.device.device_id != "pc":
                    handler._json(403, {"error": "Set up the connection from the PC browser."})
                    return True
                host = data.get("host", "")
                if not isinstance(host, str) or len(host) > 64:
                    raise ValueError("Choose a private PC address.")
                handler._json(200, self.network.enable(host))
            elif handler.path == "/api/memory/forget":
                identity = data.get("id")
                if type(identity) is not int or data.get("confirmed") is not True:
                    raise ValueError("Confirm which saved fact to forget.")
                handler._json(200, {"forgotten": self.forget(identity)})
            else:
                if data.get("confirmed") is not True:
                    raise ValueError("Confirm the server restart.")
                handler._json(200, {"restarting": True})
                timer = threading.Timer(0.25, self.restart)
                timer.daemon = True
                timer.start()
        except (ValueError, TypeError, OSError):
            handler._json(
                400,
                {
                    "error": "This change could not be applied. Check the selected "
                    "values or local address and try again."
                },
            )
        return True
