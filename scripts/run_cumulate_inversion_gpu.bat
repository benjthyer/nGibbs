@echo off
REM Cumulate-inversion GPU runs. Run from an Anaconda Prompt with your nGibbs
REM (torch + CUDA) environment active:
REM     scripts\run_cumulate_inversion_gpu.bat
REM 20 random targets per model, each with a different solid assemblage (>= 2 solids).
REM Logs and results go to "Claude outputs\cumulate_inversion_v3\gpu".
setlocal
cd /d "%~dp0.."
set PYTHONPATH=%CD%\src
set OUT=Claude outputs\cumulate_inversion_v3\gpu
if not exist "%OUT%" mkdir "%OUT%"

python -c "import torch, sys; print('python', sys.version.split()[0], 'torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')" > "%OUT%\env.txt" 2>&1
type "%OUT%\env.txt"

python -m pytest tests\unit_tests\test_cumulate_inversion.py -q > "%OUT%\pytest.log" 2>&1
type "%OUT%\pytest.log"

python -u scripts\cumulate_inversion_demo.py --model-dir src\ngibbs\engine\TrainedModels\102 --variant NoCr --nodes 32768 --steps 600 --n-assemblages 20 --device cuda --out "%OUT%\102_NoCr" > "%OUT%\102_NoCr.log" 2>&1
python -u scripts\cumulate_inversion_demo.py --model-dir src\ngibbs\engine\TrainedModels\120 --variant NoCr --nodes 32768 --steps 600 --n-assemblages 20 --device cuda --out "%OUT%\120_NoCr" > "%OUT%\120_NoCr.log" 2>&1

echo done > "%OUT%\DONE.txt"
echo All runs finished. Results in "%OUT%".
endlocal
