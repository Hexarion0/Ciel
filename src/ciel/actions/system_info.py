"""Local hardware answers using CIEL's existing telemetry backend."""

import re

from ciel.actions.base import Action, ActionContext, ActionError
from ciel.platform.system_info import get_system_snapshot


class SystemInfoAction(Action):
    """Answer bounded RAM, GPU, and CPU questions without a provider request."""

    name = "system_info"
    description = "Read current RAM use, GPU temperature, or CPU core count."

    def match(self, text: str) -> dict[str, str] | None:
        patterns = {
            "ram": r"(?:how much (?:ram|memory) am i using|(?:what is|what's) my ram usage)",
            "gpu": r"(?:(?:what is|what's) my gpu temperature|how hot is my gpu)",
            "cpu": r"how many cpu cores do i have",
        }
        for metric, pattern in patterns.items():
            if re.fullmatch(pattern, text, re.I):
                return {"metric": metric}
        return None

    def validate(self, arguments: dict[str, str]) -> None:
        if set(arguments) != {"metric"} or arguments["metric"] not in {"ram", "gpu", "cpu"}:
            raise ActionError("Choose RAM usage, GPU temperature, or CPU core count.")

    def execute(self, arguments: dict[str, str], context: ActionContext) -> str:
        snap = get_system_snapshot(gpu_timeout=min(1.5, context.remaining()))
        context.remaining()
        if arguments["metric"] == "ram":
            if snap.ram_total_gb <= 0:
                return "RAM usage is unavailable on this system."
            return (f"You're using {snap.ram_used_gb:g} of {snap.ram_total_gb:g} GB of RAM, "
                    f"about {snap.ram_percent:g} percent.")
        if arguments["metric"] == "gpu":
            if not snap.gpu.available:
                return "GPU temperature is unavailable on this system."
            return f"Your {snap.gpu.name} is at {snap.gpu.temperature_c} degrees Celsius."
        return f"Your system reports {snap.cpu_count} logical CPU cores."


ACTION = SystemInfoAction()
