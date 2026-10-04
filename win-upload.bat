@echo off
REM Safe upload: stage -> review -> commit -> rebase onto remote -> push
REM
REM   win-upload.bat                  commit message defaults to "auto update"
REM   win-upload.bat "your message"
REM
REM Deliberately NOT using `git push -f`: a force push silently discards
REM commits made from another machine (the server, another laptop).
setlocal

set "MSG=%~1"
if "%MSG%"=="" set "MSG=auto update"

for /f "delims=" %%b in ('git rev-parse --abbrev-ref HEAD 2^>nul') do set "BRANCH=%%b"
if "%BRANCH%"=="" goto :notrepo

echo ======== branch: %BRANCH% ========
git add -A
if errorlevel 1 goto :fail

git status --short
echo.

git diff --cached --quiet
if not errorlevel 1 goto :nothing

set "OK="
set /p "OK=Commit the above? [y/N] "
if /i not "%OK%"=="y" goto :aborted

git commit -m "%MSG%"
if errorlevel 1 goto :fail
goto :sync

:nothing
echo Nothing staged - working tree is clean.

:sync
echo.
echo ======== syncing with origin/%BRANCH% ========
git pull --rebase origin %BRANCH%
if errorlevel 1 goto :conflict

echo.
echo ======== pushing ========
git push origin %BRANCH%
if errorlevel 1 goto :fail

echo.
echo ======== done ========
goto :end

:aborted
echo Aborted. Nothing committed.
goto :end

:conflict
echo.
echo Rebase stopped - most likely a conflict.
echo   fix the files, then:  git add ^<file^> ^&^& git rebase --continue
echo   to undo everything:   git rebase --abort
goto :end

:notrepo
echo Not a git repository (or git is not on PATH).
goto :end

:fail
echo.
echo FAILED - see the error above. Nothing was force-pushed.

:end
endlocal
pause
