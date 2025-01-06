@echo off
setlocal enabledelayedexpansion
set /p link="YouTube download Link: "
echo 1. mp3
echo 2. video
set /p choice="choose your download type (1 or 2): "

if !choice!==1 (
    yt-dlp -x --audio-format mp3 -o "downloads\%%(uploader)s\%%(title)s.%%(ext)s" !link!
    echo Download completed
) else if !choice!==2 (
    yt-dlp -F !link!
    set /p format_code="Enter the format code: "
    yt-dlp -f !format_code! -o "downloads\%%(uploader)s\%%(title)s.%%(ext)s" !link!
    echo Download completed
)

start "" "downloads"
echo Press Enter to Exit...
pause >nul
endlocal