@echo off

rem Set the path to the Python executable
set python_executable=python

set "current_directory=%CD%"
set "launch_script=%current_directory%\create_dataset.py"

rem Set the name of the virtual environment
set venv_name=venv

rem Activate the virtual environment
echo Activating virtual environment
echo %current_directory%\%venv_name%\Scripts\activate
call %current_directory%\%venv_name%\Scripts\activate

echo calling python Script
python create_dataset.py

pause