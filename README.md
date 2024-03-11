# litvey

1. Install pdf2htmlEX

    ```bash
    wget https://github.com/pdf2htmlEX/pdf2htmlEX/releases/download/v0.18.8.rc1/pdf2htmlEX-0.18.8.rc1-master-20200630-Ubuntu-bionic-x86_64.deb
    sudo mv pdf2htmlEX-0.18.8.rc1-master-20200630-Ubuntu-bionic-x86_64.deb pdf2htmlEX.deb
    sudo apt install ./pdf2htmlEX.deb
    ```
2. Install the package requirements using 
    ```bash
    pip install -r requirements.txt

    ```

3. Run the file

    ```bash
    python3 main.py
    ```

Make sure you have pdf files in the data directory