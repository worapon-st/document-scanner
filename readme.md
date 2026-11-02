# Document Scanner

This project is designed to address the problem of needing to scan documents without a scanner.

## Get Started

1. install uv.

        pip install uv

2. clone this project and install dependencies.

        git clone
        uv sync

3. run the program

        uv run src/main.py

## Build the App (Windows)

Build one file that runs without Python:

        uv run pyinstaller --noconfirm --clean --onefile --windowed --name DocScan --paths src --specpath build src/main.py

The app is saved to `dist/DocScan.exe`. It takes a few seconds to open.

## Usage

1. Open `DocScan.exe`, or run the program from source.
2. Press IMPORT and choose a photo of a document. The app finds the page and scans it.
3. If the crop is wrong, drag the 4 white corners onto the page. The result updates when you let go.
4. Move the sliders to adjust the result.
5. Press EXPORT and save as PNG or JPEG.

## How It Works

Image functions live in `src/docscan/core.py`. The window lives in `src/docscan/app.py`.

```mermaid
flowchart TD
    import["IMPORT: load_image"] --> readable{"Image readable?"}
    readable -- No --> error["Show error"]
    readable -- Yes --> detect["find_document"]

    detect --> found{"Page found?"}
    found -- Yes --> onpage["Put corners on the page"]
    found -- No --> box["Put corners on a default box"]

    onpage --> valid{"is_valid_quad?"}
    box --> valid
    valid -- Yes --> warp["warp: flatten the page"]
    valid -- No --> keep["Keep the last scan and show a hint"]

    warp --> adjust["adjust: apply sliders"]
    adjust --> result["Show result"]
    result -- Move a slider --> adjust
    result -- Drag a corner --> valid
    keep -- Drag a corner --> valid
    result -- EXPORT --> save["save_image"]
```

## Example

1. Automation scanned mode.

/img/assets/doc_scanner_auto.png

2. Edge selection scanned mode.

/img/assets/doc_scanner_edge_select.png