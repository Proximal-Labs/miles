"""Stage A end-to-end harness: real Miles producer/buffer/store/capture, faked training.

Everything except training runs for real. The platform is a local stub that plays a
scripted multi-turn agent; the trainer is a driver that consumes batches through the
real batch query and publishes pre-made LoRA versions. Inference is either a local
fake serving pool (offline proof) or the real Miles-owned Modal serving pool.
"""
