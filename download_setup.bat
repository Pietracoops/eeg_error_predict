@echo off
setlocal

rem Get the current directory
set "current_directory=%CD%"
set "setup_directory=%current_directory%\setup"


rem Add Wget to the PATH
set "wget_path=%current_directory%\tools"
if not "%PATH:;=%" == *"%wget_path%"* (
    set "path=%wget_path%;%path%"
    echo wget added to the PATH.
) else (
    echo wget already in the PATH.
)
@REM which wget.exe

REM Set your MediaFire link
set "mediafire_link_python=https://download2278.mediafire.com/q5sfy12nezggSOmK8Ug-gvnKQ2iWQpLQMCmoOQ6-oW-yh41CwV9dq61gEm0G4Sd_KZYBckVbECw-sgbaqEZF240cK1RPNCoSpRFK_qoCRLAviMIyf6XwY9RvSswRNLLifK-t-P1f6Thf30cC86fjZeyfttqQ3GQuOdKhrJ3fnkkY/la4evkr73jcs62p/python-3.9.13-amd64.exe"
set "mediafire_link_cuda=https://download1501.mediafire.com/83ikeffov2jgowCuWcU3R7UnScKBZVt0X8RnU1dsekEj5r3wkGdSBpBq9K5F0E8rOi_vIeUaXqZi-wCpnKsC2mJOXYbD-oEHlVv2YOrdSU8Wy-sdEMg90hwcLjNW5OXc_sZmMFNo1w63SRGuhP9hd8GBQji5x1igfjjPLvjs6Vo/0mhj6zbposlt1v9/cuda_11.8.0_522.06_windows.exe"
set "mediafire_link_cudnn=https://download941.mediafire.com/ksxgilr0s3ngeI_9WYhQMeNJHcWAF80pqXtIMujk4qLJxep-6LRM8GtlQx0J0zitwTe5jmrMLxb7Z5bgXI8ujiVdIyABpAX26adRnz1CSOVulPc5ok5B3VkBdu8qDcY3TS0ojcJlG0tGd_1bVeDX7tMm4iOgefk4Ld_PN92-t1M/9u4m9ly5k2g28d6/cudnn-windows-x86_64-8.9.6.50_cuda11-archive.zip"
set "mediafire_link_zip=https://download1326.mediafire.com/8hty4q42vqngzpVBedzhAKK6ZgVDpTmltwPJyHMGHU3CD1J4NH-AnFlcg6lnREmHzu9x9FO22LljvFjWixbVzLFZtHGVhXemDU41bCHy5KZx23Xt6lwrXFjoMP4r5NTL-IYv7Ns_gcbfLvrt_V08unhTbbfsZcs_URwMTxW6CXk/gkfmr2fd55j4mbu/7z2301-x64.exe"
set "mediafire_link_data1=https://download847.mediafire.com/gvch5jnsjtvgyAjcIMxu7_hU9kkZEru7uNkSQDQRQRd8u1DE7m9Nit9I7nNWQZMcGrrR-64-KSXMZbrG-qmJDJHVS2v0vDqp7u-yXIijk7PyqNaE4ghcwVj701_UXqunLZgmMWRzKwZ41xXUp3uCqw1SIqucPWU0bS3wTx7ixCY/phmcvfyldaihi8y/flanker_data_10-4_5-_30_12_23_17_22_11.pkl"
set "mediafire_link_data2=https://download1509.mediafire.com/p669lg4hrkpgH-WvFcnaK-qYhI3MtGl3i44ww4BsQ9_WgBNte_yoJr68MW9ivN5CROX3373-RIj9fFw9LaOANBJDAznV7D2mgC9LupYRyZRzAd0i-mLBeg33tDx5fPK3Id3UtOE06E13EifFT4SJjT9z7rWNWr5CRfbTtZkDDqE/8eeyj4i0nhgud67/flanker_data_12_02_01_24_15_30_25.pkl"

REM Create the destination folder if it doesn't exist
if not exist "%setup_directory%" mkdir "%setup_directory%"

REM Download using wget (or another tool that supports MediaFire links)

wget "%mediafire_link_data1%"
wget "%mediafire_link_data2%"
cd %setup_directory%
wget "%mediafire_link_python%"
wget "%mediafire_link_cuda%"
wget "%mediafire_link_cudnn%"
wget "%mediafire_link_zip%"
REM Check if the download was successful
if %errorlevel% neq 0 (
    echo Download failed. Please check the link or try again later.
) else (
    echo Download successful!
)

endlocal