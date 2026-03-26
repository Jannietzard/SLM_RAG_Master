import time, warnings, logging
warnings.filterwarnings('ignore')
logging.disable(logging.CRITICAL)

from src.data_layer.entity_extraction import SpacyEntityPipeline

t0 = time.time()
pipe = SpacyEntityPipeline()
init_ms = (time.time()-t0)*1000

texts = [
    'Scott Derrickson is an American director born in Colorado. He is known for Doctor Strange.',
    'Ed Wood was an American filmmaker who lived in Hollywood. He directed Plan 9 from Outer Space.',
    'Marie Curie was a Polish physicist. She won the Nobel Prize in Physics in 1903.',
    'The Eiffel Tower is located in Paris, France. It was designed by Gustave Eiffel in 1887.',
]
chunk_ids = ['c1','c2','c3','c4']

t0 = time.time()
results = pipe.process_chunks_batch(texts, chunk_ids)
total_ms = (time.time()-t0)*1000

print(f'Init:        {init_ms:.0f}ms')
print(f'Extraktion:  {total_ms:.0f}ms gesamt | {total_ms/len(texts):.1f}ms/chunk')
print()
for r in results:
    print(f'[{r.chunk_id}] {len(r.entities)} Entities ({r.extraction_time_ms:.1f}ms):')
    for e in r.entities:
        print(f'  {repr(e.name):28s} {e.entity_type:14s} conf={e.confidence:.2f}')
    print()
