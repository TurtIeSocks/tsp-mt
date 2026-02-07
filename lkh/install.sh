version="3.0.13"

echo Installing LKH v${version}

if [ ! -d "${version}" ]; then
        curl -L "http://akira.ruc.dk/~keld/research/LKH-3/LKH-${version}.tgz" -o lkh.tgz
        tar xvfz lkh.tgz
        cd "LKH-${version}"
        make
fi
