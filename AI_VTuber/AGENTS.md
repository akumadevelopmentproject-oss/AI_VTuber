# Luna security boundaries

Read docs/PRIVATSPHAERE.md before changing chat, tools, memory or controls.
PublicChat is the ONLY audience entry point. Never forward platform text to the
private AI, memory handlers, local terminal input, or shell/keyboard controls.
The default startup must be public. Owner control is local keyboard input, not
names, claims, roles typed in chat, or speaker recognition. Keep private/public
histories and prompts separate. Public TTS must not load private replacements.
Do not send messages to real viewers in tests. Use mock Ollama responses and
synthetic secrets; run python -m unittest discover -s tests -v.
Never commit local memories, keys, recordings, .env, or __pycache__.
