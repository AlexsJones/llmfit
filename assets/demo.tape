# README demo for llmfit. Render with: scripts/record_demo.sh
#
# Story (~30s): default ranked list -> search for a newly added model that is
# too big for this machine -> simulate bigger hardware and watch it fit ->
# detail view -> plan view -> back to the list, cycle themes.
#
# Frames are written as PNGs and encoded by the script rather than by VHS, so
# the GIF can be quantised without dithering (flat TUI colours need none, and
# dithering is what makes terminal GIFs look grainy).

Output demo-frames/

Require llmfit

Set Shell "bash"
Set FontSize 14
Set Width 1560
Set Height 860
Set Padding 12
Set Framerate 24
Set TypingSpeed 80ms
Set Theme "Catppuccin Mocha"
Set CursorBlink false

# A throwaway config dir, so the recording shows default sort, filters and
# theme instead of whatever the person recording has saved.
Hide
Type "export XDG_CONFIG_HOME=$(mktemp -d) PS1='$ ' && clear" Enter
Sleep 500ms
Show

Type "llmfit" Sleep 500ms Enter
Sleep 3.4s

# 1. Find a newly added model: too big for this machine.
Type "/" Sleep 500ms
Type "zai-org/glm-5.3" Sleep 1.4s
Enter Sleep 2.4s

# 2. Simulate bigger hardware (1 TB RAM, 640 GB VRAM) and watch it fit.
Type "S" Sleep 1.3s
Ctrl+U Type "1024" Sleep 500ms
Tab Sleep 400ms
Ctrl+U Type "640" Sleep 1.1s
Enter Sleep 3.2s

# 3. Detail, then plan, on the simulated machine.
Enter Sleep 3.6s
Enter Sleep 500ms
Type "p" Sleep 3.6s
Escape Sleep 600ms

# 4. Back to the full list, cycle themes.
Type "/" Sleep 200ms Ctrl+U Sleep 200ms Enter Sleep 1.4s
Type "t" Sleep 1.3s
Type "t" Sleep 2.4s
