::~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
:: RUN CUDATemplateClass
::~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
@echo off
echo ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
echo RUN CUDATemplateClass
echo ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

echo ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 
echo RUN func1
echo ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 
CALL timecmd x64\Release\mainTemplateClass.exe -debug=1 -mode="func1" -coutMode=1 -outputCSV=1 -numFilas=1200 -fileOut="data\\resultadofunc1.csv"

echo ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 
echo RUN func2
echo ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 
CALL timecmd x64\Release\mainTemplateClass.exe -debug=1 -mode="func2" -coutMode=1 -outputCSV=1 -numFilas=800 -fileOut="data\\resultadofunc2.csv"
