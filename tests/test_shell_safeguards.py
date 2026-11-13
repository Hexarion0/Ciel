"""Safeguard parsing: operators without spaces, home-relative globs, and file-reading flags."""

import pytest

from ciel import shell_tool


@pytest.mark.parametrize("command", [
    "echo hi;rm -rf ~", "true&&rm -rf ~", "ls|rm -rf ~", "(rm -rf ~)", "echo $(rm -rf ~)",
    "echo `rm -rf ~`", "cd /tmp;rm -rf /",
    # Commands run from the home folder, so these remove everything in it.
    "rm -rf *", "rm -rf .", "rm -rf ./", "rm -rf ./*", "rm -rf ..", "rm -rf .*",
])
def test_destructive_commands_are_recognized_without_spaces_or_from_home(command):
    assert shell_tool.is_destructive(command)


@pytest.mark.parametrize("command", [
    "rm -rf ./build", "rm -rf ~/build", "echo a;echo b", "ls -la ~|wc -l", "rm -rf build/*",
])
def test_routine_commands_with_operators_stay_routine(command):
    assert not shell_tool.is_destructive(command)


@pytest.mark.parametrize("command", [
    "du --files0-from=/etc/hostname", "du --files0=/etc/hostname", "du --f=/etc/hostname",
    "du -X ~/.ssh/id_rsa /tmp", "du --exclude-from=notes.txt", "du --max=1",
])
def test_du_flags_that_read_files_need_approval(command):
    assert not shell_tool.is_safe_listed(command)


@pytest.mark.parametrize("command", [
    "du", "du -sh ~/Downloads", "du -h --max-depth=1 /var", "du -hd1 ~", "du -d 2 .",
    "du -sch ~/Videos ~/Music", "du --apparent-size --summarize ~",
])
def test_ordinary_disk_usage_stays_on_the_read_only_list(command):
    assert shell_tool.is_safe_listed(command)
