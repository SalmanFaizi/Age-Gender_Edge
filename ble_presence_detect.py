import asyncio
from bleak import BleakScanner

# The exact MAC address of your iTag (converted to lowercase for reliable comparison)
TARGET_MAC = "cc:9b:52:17:34:4b"

def detection_callback(device, advertisement_data):
    """
    Runs automatically every time a BLE device broadcasts.
    Filters specifically for your target iTag.
    """
    # Check if this broadcast matches your iTag's MAC address
    if device.address.lower() == TARGET_MAC:
        name = device.name if device.name else "iTag"
        rssi = advertisement_data.rssi
        
        # Print the live tracking status
        print(f"?? Found target iTag! | MAC: {device.address} | Name: {name} | RSSI: {rssi} dBm")

async def main():
    print(f"?? Monitoring range for iTag [{TARGET_MAC.upper()}]...")
    print("Press Ctrl+C to stop.\n")
    
    # Initialize scanner with the custom filter callback
    scanner = BleakScanner(detection_callback)
    
    # Start continuous background scanning
    await scanner.start()
    
    try:
        # Keep the script running to listen for advertisements
        while True:
            await asyncio.sleep(1)
    finally:
        # Stop scanner gracefully on exit
        await scanner.stop()

try:
    asyncio.run(main())
except KeyboardInterrupt:
    print("\nMonitoring stopped.")
