@echo off

>nul 2>&1 "%SYSTEMROOT%\system32\cacls.exe" "%SYSTEMROOT%\system32\config\system"

if '%errorlevel%' NEQ '0' ( goto UACPrompt ) else ( goto GotAdmin )

:UACPrompt
	echo Set UAC = CreateObject^("Shell.Application"^) > "%temp%\getadmin.vbs"
	echo UAC.ShellExecute "%~s0", "", "", "runas", 1 >> "%temp%\getadmin.vbs"
	"%temp%\getadmin.vbs"
	exit /B

:GotAdmin
	if exist "%temp%\getadmin.vbs" ( del "%temp%\getadmin.vbs" )
	cd /d "%~dp0"
	set CARGO_PROFILE_RELEASE_DEBUG=true
	cargo flamegraph --unit-test
