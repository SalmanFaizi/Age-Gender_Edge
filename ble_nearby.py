import asyncio
from bleak import BleakScanner

def detection_callback(device, advertisement_data):
    """
    This function runs automatically every time a BLE advertisement is detected.
    advertisement_data cleanly provides the RSSI value.
    """
    # Handle devices that don't broadcast a friendly name
    name = device.name if device.name else "Unknown Device"
    
    # Safely get the RSSI from the advertisement data
    rssi = advertisement_data.rssi
    
    # Print the MAC address, Name, and RSSI
    print(f"?? MAC: {device.address} | Name: {name} | RSSI: {rssi} dBm")

async def main():
    print("?? Scanning for all nearby BLE devices... Press Ctrl+C to stop.\n")
    
    # Register the callback function into the scanner
    scanner = BleakScanner(detection_callback)
    
    # Start scanning continuously
    await scanner.start()
    
    try:
        # Keep scanning indefinitely until interrupted
        while True:
            await asyncio.sleep(1)
    finally:
        # Ensure the scanner stops properly when stopping the script
        await scanner.stop()

try:
    asyncio.run(main())
except KeyboardInterrupt:
    print("\nScanner stopped.")
