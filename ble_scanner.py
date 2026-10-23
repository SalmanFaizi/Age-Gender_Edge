import asyncio
from bleak import BleakScanner

# !!! REPLACE THIS WITH YOUR ITAG's MAC ADDRESS (Use lowercase) !!!
ITAG_MAC_ADDRESS = "CC:9B:52:17:34:4B" 

async def main():
    print("Scanning for your iTag... Press Ctrl+C to stop.")
    
    while True:
        # Scan for 3 seconds
        devices = await BleakScanner.discover(timeout=2.0)
        
        found = False
        for device in devices:
            if device.address.lower() == ITAG_MAC_ADDRESS:
                # RSSI is the signal strength (e.g., -50 is very close, -90 is far away)
                print(f"? iTag is IN RANGE! Signal Strength (RSSI): {device.rssi} dBm")
                found = True
                break
                
        if not found:
            print("? iTag is OUT OF RANGE.")
            
        # Wait 2 seconds before scanning again
        await asyncio.sleep(2)

try:
    asyncio.run(main())
except KeyboardInterrupt:
    print("\nScanner stopped.")
