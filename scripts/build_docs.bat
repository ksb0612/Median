@ECHO OFF
REM Build Sphinx documentation for Windows

SETLOCAL

ECHO Building Sphinx documentation...

REM Check if we're in the right directory
IF NOT EXIST "pyproject.toml" (
    ECHO Error: Must be run from project root directory
    EXIT /B 1
)

REM Check if Poetry is installed
WHERE poetry >NUL 2>NUL
IF %ERRORLEVEL% NEQ 0 (
    ECHO Error: Poetry is not installed
    ECHO Install Poetry: https://python-poetry.org/docs/#installation
    EXIT /B 1
)

REM Check if docs dependencies are installed
ECHO Checking documentation dependencies...
poetry run python -c "import sphinx" >NUL 2>NUL
IF %ERRORLEVEL% NEQ 0 (
    ECHO Installing documentation dependencies...
    poetry install --with docs
)

REM Clean previous build
ECHO Cleaning previous build...
CD docs
IF EXIST "build" (
    RMDIR /S /Q build
)

REM Build HTML documentation
ECHO Building HTML documentation...
poetry run sphinx-build -W -b html source build\html

REM Check if build was successful
IF %ERRORLEVEL% EQU 0 (
    ECHO.
    ECHO ✅ Documentation built successfully!
    ECHO 📄 Open docs\build\html\index.html to view
    ECHO.

    REM Get full path
    SET DOCS_PATH=%CD%\build\html\index.html
    ECHO File location: %DOCS_PATH%

    REM Open in browser
    ECHO Opening in browser...
    START "" "%DOCS_PATH%"
) ELSE (
    ECHO.
    ECHO ❌ Documentation build failed
    CD ..
    EXIT /B 1
)

CD ..
ENDLOCAL
