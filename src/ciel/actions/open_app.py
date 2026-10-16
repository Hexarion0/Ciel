"""Launch an allowlisted desktop executable without shell interpolation."""

import re
import shutil
import subprocess
import sys
import threading

from ciel.actions.base import Action, ActionContext, ActionError

APPLICATIONS = {
    "discord": ("discord", "vesktop"),
    "vesktop": ("vesktop",),
    "firefox": ("firefox",),
    "chromium": ("chromium", "chromium-browser"),
    "steam": ("steam",),
    "code": ("code", "codium"),
    "visual studio code": ("code", "codium"),
}


class OpenAppAction(Action):
    """Start known Linux applications without accepting paths, flags, or commands."""

    name = "open_app"
    description = "Launch Discord, Vesktop, Firefox, Chromium, Steam, or Visual Studio Code."
    mutating = True

    @staticmethod
    def _normalize_app(text: str) -> str:
        """Normalize exact names and letter-by-letter spelling, never phonetic guesses."""
        name = " ".join(text.casefold().split())
        name = re.sub(r"^the\s+", "", name)
        name = re.sub(r"\s+app(?:lication)?$", "", name)
        if re.fullmatch(r"[a-z](?:[ -]+[a-z])+", name):
            name = re.sub(r"[ -]+", "", name)
        return name

    def match(self, text: str) -> dict[str, str] | None:
        match = re.fullmatch(
            r"(?:(?:can|could|would|will) you\s+)?(?:please\s+)?"
            r"(?:open|launch|start)\s+(.{1,120}?)"
            r"(?:\s+for me)?(?:[, ]+please)?",
            text, re.I,
        )
        return {"app": self._normalize_app(match.group(1))} if match else None

    def match_with_context(
        self, text: str, recent_requests: list[str],
    ) -> dict[str, str] | None:
        """Resolve 'open it' only from a recent exact app name or spelling correction."""
        arguments = self.match(text)
        if arguments is None or arguments["app"] not in {"it", "that", "app", "that app"}:
            return arguments
        for request in reversed(recent_requests):
            previous = self.match(request)
            if previous is not None:
                name = previous["app"]
                if name in {"it", "that", "app", "that app"}:
                    continue
            else:
                correction = re.fullmatch(
                    r"(?:no[, .]+){0,3}(?:i meant\s+)?(.{1,120})",
                    request.strip().rstrip(".!?"), re.I,
                )
                name = self._normalize_app(correction.group(1)) if correction else ""
            # An unrelated/ambiguous intervening turn ends the reference chain.
            if name in APPLICATIONS:
                return {"app": name}
            break
        return arguments

    def validate(self, arguments: dict[str, str]) -> None:
        if arguments.get("app") in {"it", "that", "app", "that app"}:
            raise ActionError("Which application should I open? Please say its name.")
        if set(arguments) != {"app"} or arguments["app"] not in APPLICATIONS:
            raise ActionError(
                "I can open Discord, Vesktop, Firefox, Chromium, Steam, or Visual Studio Code. "
                "Please name one of those applications."
            )

    def execute(self, arguments: dict[str, str], context: ActionContext) -> str:
        if sys.platform != "linux":
            return "Application launching currently supports Linux."
        app = arguments["app"]
        executable = next((path for name in APPLICATIONS[app]
                           if (path := shutil.which(name))), None)
        if executable is None:
            return f"I couldn't find {app} installed on your executable search path."
        context.remaining()
        process = subprocess.Popen(
            [executable], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL, start_new_session=True,
        )
        # Reap the child on exit without blocking the voice loop or terminating the app.
        threading.Thread(target=process.wait, daemon=True, name="app-reaper").start()
        return f"I've sent the launch request for {app}."


ACTION = OpenAppAction()
