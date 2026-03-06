#!/bin/bash
pip install numpy
pip install cython
pip install icecream
pip install scipy
pip install paoflow

cd minijclsquant/cython_modules/
rm blas_funs.c
rm blas_funs.html
rm blas_funs.cpython-310-x86_64-linux-gnu.so
rm Extra_funs.c
rm Extra_funs.cpp       
rm Extra_funs.html
rm -r build
python Extra_funs_setup.py build_ext --inplace
python blas_funs.py build_ext --inplace
cd ../
pwd
read -p "Do you want to use  CPU or GPU? " choice
choice=$(echo "$choice" | tr '[:upper:]' '[:lower:]')

if [[ "$choice" == "cpu" ]]; then
    pwd
    sed -i '/from minijclsquant\.observables_gpu import \*/d' __init__.py
    rm -r ./cuda_cython
    pwd
    echo "Line removed for CPU mode."
    
elif [[ "$choice" == "gpu" ]]; then
    echo "Using GPU."
    read -p "Enter the compute capability number (e.g., 89 for sm_89) you can check in https://developer.nvidia.com/cuda-gpus : " SM
    cd ./cuda_cython/cuda
    nvcc -Xcompiler -fPIC -shared -o libobv.so obv.cu modifiers.cu recursions.cu blas.cu -arch=sm_$SM
    read -p "Enter the full path to your Conda environment's lib folder suggested ~/miniconda3/lib/: " LIB_PATH
    cp libobv.so "$LIB_PATH"
    cp ./libobv.so ../../

    cd ../
    rm -r build
    rm obv.cpp
    rm obv_gpu.cpython-310-x86_64-linux-gnu.so 

    read -p "Enter your CUDA version (e.g., 12 or 13): " CUDA_VERSION

    # Choose setup file depending on CUDA version
    if [[ "$CUDA_VERSION" == "13" ]]; then
        echo "Detected CUDA 13. Using obv_setup_13.py"
        python obv_setup_13.py build_ext --inplace
    else
        echo "Using standard CUDA setup (obv_setup.py)"
        python obv_setup.py build_ext --inplace
    fi

else
    echo "Invalid choice. Please type 'CPU' or 'GPU'."
fi
pwd
cd ../


pip install -e .

python3 -c "import minijclsquant; print('minijclsquant imported successfully')"
if [[ "$choice" == "cpu" ]]; then
	python3 dos_test_only_cpu.py
elif [[ "$choice" == "gpu" ]]; then
	python3 dos_test_both.py
fi
