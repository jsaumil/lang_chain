from langchain.text_splitter import RecursiveCharacterTextSplitter, Language

text = """
class TwilioGoogleSTTIntegration:
    def __init__(self):
        self.client = speech.SpeechClient()
        self.recognize_stream = None

    async def handle_message(self, websocket, message):
        msg = json.loads(message)
        
        if msg['event'] == 'connected':
            print('Twilio media stream connected.')
            # Configure Google STT
            config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.MULAW,
                sample_rate_hertz=8000,
                alternative_language_codes=['en-US', 'es-ES', 'fr-FR', 'de-DE', 'hi-IN'],
                enable_automatic_punctuation=True,
                profanity_filter=False
            )
            
            streaming_config = speech.StreamingRecognitionConfig(
                config=config,
                interim_results=False
            )
            
            # Create generator for audio chunks
            self.audio_generator = self.audio_chunk_generator()
            
            # Start streaming recognition
            requests = (speech.StreamingRecognizeRequest(audio_content=chunk)
                        for chunk in self.audio_generator)
            
            self.recognize_stream = self.client.streaming_recognize(
                streaming_config,
                requests
            )
            
            # Start processing responses in background
            asyncio.create_task(self.process_responses())
            
        elif msg['event'] == 'start':
            print('Twilio has started sending media.')
            
        elif msg['event'] == 'media':
            if self.recognize_stream:
                # Queue the audio data for processing
                await self.audio_queue.put(msg['media']['payload'])
                
        elif msg['event'] == 'stop':
            print('Twilio has stopped sending media.')
            if self.recognize_stream:
                self.recognize_stream.cancel()
            self.recognize_stream = None

    async def audio_chunk_generator(self):
        while True:
            chunk = await self.audio_queue.get()
            if chunk is None:  # Sentinel value to stop
                break
            yield chunk

    async def process_responses(self):
        try:
            async for response in self.recognize_stream:
                for result in response.results:
                    if result.is_final:
                        transcript = result.alternatives[0].transcript
                        language = result.language_code
                        print(f'Google STT Transcription ({language}): {transcript}')
        except Exception as e:
            print(f'Error processing responses: {e}')

async def handler(websocket, path):
    print('New WebSocket connection established.')
    integration = TwilioGoogleSTTIntegration()
    
    try:
        async for message in websocket:
            await integration.handle_message(websocket, message)
    except websockets.exceptions.ConnectionClosed:
        print('WebSocket connection closed.')
        if integration.recognize_stream:
            integration.recognize_stream.cancel()

async def main():
    server = await websockets.serve(handler, "localhost", 8080)
    print(f'WebSocket server started on port 8080')
    await server.wait_closed()

if __name__ == "__main__":
    asyncio.run(main())
"""

splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size = 400,
    chunk_overlap = 0,
)

chunks = splitter.split_text(text)

print(len(chunks))
print(chunks[0])