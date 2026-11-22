// Fixed-capacity ring buffer preventing JS GC spikes at 60 fps on 4 GB VRAM laptops.

export interface HudHistorySample {
  seq: number;
  simTimeS: number;
  vTermV: number;
  socPct: number;
  cbvfMinV: number;
  aglM: number;
}

export class HudRingStore {
  private readonly buffer: (HudHistorySample | undefined)[];
  private head = 0;
  private count = 0;

  constructor(public readonly capacity = 300) {
    if (capacity <= 0) {
      throw new Error("HudRingStore capacity must be positive");
    }
    this.buffer = new Array<HudHistorySample | undefined>(capacity);
  }

  push(sample: HudHistorySample): void {
    this.buffer[this.head] = sample;
    this.head = (this.head + 1) % this.capacity;
    if (this.count < this.capacity) {
      this.count++;
    }
  }

  get length(): number {
    return this.count;
  }

  latest(): HudHistorySample | undefined {
    if (this.count === 0) return undefined;
    const idx = (this.head - 1 + this.capacity) % this.capacity;
    return this.buffer[idx];
  }

  oldest(): HudHistorySample | undefined {
    if (this.count === 0) return undefined;
    const idx = (this.head - this.count + this.capacity) % this.capacity;
    return this.buffer[idx];
  }

  clear(): void {
    this.head = 0;
    this.count = 0;
  }
}
