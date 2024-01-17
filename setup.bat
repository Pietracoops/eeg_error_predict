@echo off
setlocal enabledelayedexpansion

rem Get the current directory
set "current_directory=%CD%"
set "setup_directory=%current_directory%\setup"

rem Check if the setup directory exists
if not exist "%setup_directory%" (
    echo Setup directory not found! Creating it...
    mkdir %setup_directory%
    call download_setup.bat
)

rem Find CUDA installer in the setup directory
for %%I in ("%setup_directory%\cuda*.exe") do set "cuda_installer_path=%%I"

if not defined cuda_installer_path (
    echo Error: CUDA installer not found in the setup directory!
    exit /b 1
)

rem Find cuDNN ZIP file in the setup directory
for %%I in ("%setup_directory%\cudnn*.zip") do set "cudnn_zip_path=%%I"
for %%I in ("!cudnn_zip_path!") do set "archivename=%%~nI"

if not defined cudnn_zip_path (
    echo Error: cuDNN ZIP file not found in the setup directory!
    exit /b 1
)

rem Install CUDA Toolkit
where nvcc >nul 2>nul
if %errorlevel% neq 0 (
    echo Installing CUDA Toolkit...
    start /wait %cuda_installer_path% -s
) else (
    echo CUDA is already installed.
)

rem Install Python 3.9
where python >nul 2>nul
if %errorlevel% neq 0 (
    set "python_installer_path=%current_directory%\setup\python-3.9.13-amd64.exe"
    echo Installing Python 3.9...
    start /wait %python_installer_path% /quiet InstallAllUsers=1 PrependPath=1
) else (
    echo Python is already installed.
)

rem Add your Python distribution to the PATH
set "python_path=C:\Program Files\Python39"
if not "%PATH:;=%" == *"%python_path%"* (
    set "PATH=!python_path!;%PATH%"
    echo Your Python distribution added to the PATH.
) else (
    echo Your Python distribution is already in the PATH.
)

rem Check if 7zip is installed, and if not, install it
if not exist "C:\Program Files\7-Zip\7z.exe" (
    echo 7zip is not installed. Installing 7zip...
    REM Modify the path to the 7zip installer if needed
    set "zip_installer_path=%current_directory%\setup\7z2301-x64.exe"
    start /wait %zip_installer_path% /S
) else (
    echo 7zip is already installed.
)

rem Add 7zip binaries to the PATH
set "zip_bin_path=C:\Program Files\7-Zip"
if not "%PATH:;=%" == *"%zip_bin_path%"* (
    set "PATH=%zip_bin_path%;%PATH%"
    echo 7zip binaries added to the PATH.
) else (
    echo 7zip binaries are already in the PATH.
)

rem Check if xcopy is in the PATH, and if not, add C:\Windows\System32 to the PATH
where xcopy >nul 2>nul
if %errorlevel% neq 0 (
    set PATH "%PATH%;C:\Windows\System32" /M
    echo C:\Windows\System32 added to the PATH.
) else (
    echo xcopy is already in the PATH.
)


if not exist "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin\cudnn_adv_infer64_8.dll" (
    rem Extract cuDNN
    echo Extracting cuDNN...
    mkdir C:\cuda
    7z x -oC:\cuda %cudnn_zip_path%

    rem Copy cuDNN bin contents to CUDA bin directory
    echo Copying cuDNN bin contents to CUDA bin directory...
    xcopy /Y /E C:\cuda\!archivename!\bin\* "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin"

    rem Copy cuDNN lib/x64 contents to CUDA lib/x64 directory
    echo Copying cuDNN lib/x64 contents to CUDA lib/x64 directory...
    xcopy /Y /E C:\cuda\!archivename!\lib\x64\* "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\lib\x64"

    rem Copy cuDNN include contents to CUDA include directory
    echo Copying cuDNN include contents to CUDA include directory...
    xcopy /Y /E C:\cuda\!archivename!\include\* "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\include"

    rem Delete contents of the C:\cuda folder
    @REM echo Deleting contents of C:\cuda...
    @REM rmdir /S /Q C:\cuda
) else (
    echo CUDNN is already setup.
)

rem Set environment variables
echo Setting environment variables...
set CUDA_PATH "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8" /M
set PATH "%CUDA_PATH%\bin;%PATH%" /M
set CUDA_LIB_PATH "%CUDA_PATH%\lib\x64" /M
set PATH "%CUDA_LIB_PATH%;%PATH%" /M

rem Display completion message
echo CUDA and cuDNN installation completed.
echo Validate using "nvidia-smi" command on command-line

rem ================== Setup python environment ===================
rmdir /S /Q "%current_directory%\venv"
rem Set the path to the Python executable
set python_executable=python

rem Set the name of the virtual environment
set venv_name=venv

rem Set the path to the requirements.txt file
set requirements_file=requirements.txt

rem Create a virtual environment
%python_executable% -m venv %venv_name%

rem Activate the virtual environment
call %venv_name%\Scripts\activate

rem Install dependencies from requirements.txt
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r %requirements_file%

echo Virtual environment created and activated. Dependencies installed.
echo To deactivate the virtual environment, run: deactivate


rem call this to validate environment
@REM nvidia-smi

endlocal
pause