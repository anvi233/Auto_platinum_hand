#NoEnv
SetCapsLockState, AlwaysOff
WinGet, remote_id, List, ahk_exe chiaki-ng.exe
if (remote_id = 0)
    WinGet, remote_id, List, ahk_exe chiaki.exe
if (remote_id = 0) {
    MsgBox, 16, Error, Chiaki not find
    ExitApp
}
global targetID := remote_id1
global Playing := 0
ToolTip, AutoPlatinum: Ready! (Press CapsLock to Play), 10, 10

$CapsLock::
    Playing := !Playing
    if (Playing) {
        ToolTip, AutoPlatinum: PLAYING..., 10, 10
        SetTimer, PlayMacro, -1
    } else {
        ToolTip
        Reload
    }
return

PlayMacro:
DllCall("Sleep", "Uint", 500)
DllCall("Sleep", "Uint", 50)
DllCall("Sleep", "Uint", 445)
ControlSend, , {Up down}, ahk_id %targetID%
DllCall("Sleep", "Uint", 171)
ControlSend, , {Up up}, ahk_id %targetID%
DllCall("Sleep", "Uint", 1494)
ControlSend, , {Enter down}, ahk_id %targetID%
DllCall("Sleep", "Uint", 50)
ControlSend, , {Enter up}, ahk_id %targetID%
DllCall("Sleep", "Uint", 561)
ControlSend, , {Enter down}, ahk_id %targetID%
DllCall("Sleep", "Uint", 50)
ControlSend, , {Enter up}, ahk_id %targetID%
DllCall("Sleep", "Uint", 0)
ControlSend, , {Right down}, ahk_id %targetID%
ControlSend, , {Down down}, ahk_id %targetID%
DllCall("Sleep", "Uint", 529)
ControlSend, , {Right up}, ahk_id %targetID%
ControlSend, , {Down up}, ahk_id %targetID%
DllCall("Sleep", "Uint", 129)
ControlSend, , {Enter down}, ahk_id %targetID%
DllCall("Sleep", "Uint", 50)
ControlSend, , {Enter up}, ahk_id %targetID%
DllCall("Sleep", "Uint", 55)
ControlSend, , {Up down}, ahk_id %targetID%
DllCall("Sleep", "Uint", 234)
ControlSend, , {Up up}, ahk_id %targetID%
DllCall("Sleep", "Uint", 900)
ControlSend, , {Enter down}, ahk_id %targetID%
DllCall("Sleep", "Uint", 50)
ControlSend, , {Enter up}, ahk_id %targetID%
DllCall("Sleep", "Uint", 621)
ControlSend, , {Right down}, ahk_id %targetID%
ControlSend, , {Up down}, ahk_id %targetID%
DllCall("Sleep", "Uint", 249)
ControlSend, , {Right up}, ahk_id %targetID%
ControlSend, , {Up up}, ahk_id %targetID%
DllCall("Sleep", "Uint", 2621)
ControlSend, , {Enter down}, ahk_id %targetID%
DllCall("Sleep", "Uint", 50)
ControlSend, , {Enter up}, ahk_id %targetID%
DllCall("Sleep", "Uint", 127)
ControlSend, , {Enter down}, ahk_id %targetID%
DllCall("Sleep", "Uint", 50)
ControlSend, , {Enter up}, ahk_id %targetID%
DllCall("Sleep", "Uint", 2036)
ControlSend, , {Up down}, ahk_id %targetID%
DllCall("Sleep", "Uint", 264)
ControlSend, , {Up up}, ahk_id %targetID%
DllCall("Sleep", "Uint", 673)
ControlSend, , {Enter down}, ahk_id %targetID%
DllCall("Sleep", "Uint", 50)
ControlSend, , {Enter up}, ahk_id %targetID%
DllCall("Sleep", "Uint", 580)
ControlSend, , {Enter down}, ahk_id %targetID%
DllCall("Sleep", "Uint", 50)
ControlSend, , {Enter up}, ahk_id %targetID%
DllCall("Sleep", "Uint", 39)
ControlSend, , {Right down}, ahk_id %targetID%
ControlSend, , {Up down}, ahk_id %targetID%
DllCall("Sleep", "Uint", 217)
ControlSend, , {Right up}, ahk_id %targetID%
ControlSend, , {Up up}, ahk_id %targetID%
DllCall("Sleep", "Uint", 699)
ControlSend, , {Enter down}, ahk_id %targetID%
DllCall("Sleep", "Uint", 50)
ControlSend, , {Enter up}, ahk_id %targetID%
DllCall("Sleep", "Uint", 2264)
ControlSend, , {Enter down}, ahk_id %targetID%
DllCall("Sleep", "Uint", 50)
ControlSend, , {Enter up}, ahk_id %targetID%
DllCall("Sleep", "Uint", 837)
ControlSend, , {Enter down}, ahk_id %targetID%
DllCall("Sleep", "Uint", 50)
ControlSend, , {Enter up}, ahk_id %targetID%
DllCall("Sleep", "Uint", 848)
ControlSend, , {Enter down}, ahk_id %targetID%
DllCall("Sleep", "Uint", 50)
ControlSend, , {Enter up}, ahk_id %targetID%
DllCall("Sleep", "Uint", 892)
ControlSend, , {Right down}, ahk_id %targetID%
ControlSend, , {Up down}, ahk_id %targetID%
DllCall("Sleep", "Uint", 791)
ControlSend, , {Right up}, ahk_id %targetID%
ControlSend, , {Up up}, ahk_id %targetID%
DllCall("Sleep", "Uint", 763)
ControlSend, , {Enter down}, ahk_id %targetID%
DllCall("Sleep", "Uint", 50)
ControlSend, , {Enter up}, ahk_id %targetID%
DllCall("Sleep", "Uint", 554)
ControlSend, , {Left down}, ahk_id %targetID%
ControlSend, , {Down down}, ahk_id %targetID%
DllCall("Sleep", "Uint", 563)
ControlSend, , {Left up}, ahk_id %targetID%
ControlSend, , {Down up}, ahk_id %targetID%
DllCall("Sleep", "Uint", 1234)
ControlSend, , {Right down}, ahk_id %targetID%
ControlSend, , {Up down}, ahk_id %targetID%
DllCall("Sleep", "Uint", 481)
ControlSend, , {Right up}, ahk_id %targetID%
ControlSend, , {Up up}, ahk_id %targetID%
DllCall("Sleep", "Uint", 929)
ControlSend, , {Enter down}, ahk_id %targetID%
DllCall("Sleep", "Uint", 50)
ControlSend, , {Enter up}, ahk_id %targetID%
DllCall("Sleep", "Uint", 513)
ControlSend, , {Left down}, ahk_id %targetID%
DllCall("Sleep", "Uint", 1030)
ControlSend, , {Left up}, ahk_id %targetID%
DllCall("Sleep", "Uint", 1080)
ControlSend, , {Enter down}, ahk_id %targetID%
DllCall("Sleep", "Uint", 50)
ControlSend, , {Enter up}, ahk_id %targetID%
DllCall("Sleep", "Uint", 512)
ControlSend, , {Right down}, ahk_id %targetID%
ControlSend, , {Down down}, ahk_id %targetID%
DllCall("Sleep", "Uint", 717)
ControlSend, , {Right up}, ahk_id %targetID%
ControlSend, , {Down up}, ahk_id %targetID%
DllCall("Sleep", "Uint", 2144)
ControlSend, , {Right down}, ahk_id %targetID%
DllCall("Sleep", "Uint", 370)
ControlSend, , {Right up}, ahk_id %targetID%
DllCall("Sleep", "Uint", 2013)
ControlSend, , {Right down}, ahk_id %targetID%
ControlSend, , {Up down}, ahk_id %targetID%
DllCall("Sleep", "Uint", 323)
ControlSend, , {Right up}, ahk_id %targetID%
ControlSend, , {Up up}, ahk_id %targetID%
DllCall("Sleep", "Uint", 797)
ControlSend, , {Enter down}, ahk_id %targetID%
DllCall("Sleep", "Uint", 50)
ControlSend, , {Enter up}, ahk_id %targetID%
DllCall("Sleep", "Uint", 514)
ControlSend, , {Left down}, ahk_id %targetID%
ControlSend, , {Down down}, ahk_id %targetID%
DllCall("Sleep", "Uint", 1141)
ControlSend, , {Left up}, ahk_id %targetID%
ControlSend, , {Down up}, ahk_id %targetID%
DllCall("Sleep", "Uint", 698)
ControlSend, , {Enter down}, ahk_id %targetID%
DllCall("Sleep", "Uint", 50)
ControlSend, , {Enter up}, ahk_id %targetID%
DllCall("Sleep", "Uint", 56)
ControlSend, , {Enter down}, ahk_id %targetID%
DllCall("Sleep", "Uint", 50)
ControlSend, , {Enter up}, ahk_id %targetID%
DllCall("Sleep", "Uint", 232)
ControlSend, , {Right down}, ahk_id %targetID%
ControlSend, , {Up down}, ahk_id %targetID%
DllCall("Sleep", "Uint", 94)
ControlSend, , {Right up}, ahk_id %targetID%
ControlSend, , {Up up}, ahk_id %targetID%
ToolTip, AutoPlatinum: FINISHED!, 10, 10
Playing := 0
return

^Esc::ExitApp
