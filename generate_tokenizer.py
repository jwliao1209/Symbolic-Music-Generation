from pathlib import Path
from miditok import REMI, TokenizerConfig


if __name__ == "__main__":
    config = TokenizerConfig(
        num_velocities=16,
        use_chords=True,
        use_programs=True,
        use_tempos=True,
    )
    tokenizer = REMI(config)
    tokenizer.train(
        vocab_size=30000,
        files_paths=list(Path("Pop1K7", "midi_analyzed").glob("**/*.mid")),
    )
    tokenizer.save(Path("tokenizers", f"{tokenizer.__class__.__name__.lower()}.json"))
