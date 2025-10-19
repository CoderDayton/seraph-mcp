# ⚡ Seraph MCP — 60-Second Quickstart

**Current Status:** ✅ Production-Ready (All tests passing, linter clean)

---

## 🚀 Start the Server (Choose One)

### **Option 1: Manual Test (See Logs)**
```bash
cd /home/malu/.projects/seraph-mcp
uv run seraph-mcp
```
**What You'll See:**
```
╭──────────────────────────────────────────────────╮
│  FastMCP 2.0                                     │
│  Server name: Seraph MCP - AI Optimization...   │
│  Transport: STDIO                                │
╰──────────────────────────────────────────────────╯
INFO - Initializing Seraph MCP server...
INFO - Provider: openai_compatible (chutesai/Ling-1T-FP8)
INFO - Cache backend: memory
INFO - Context optimization: enabled (hybrid mode)
```

### **Option 2: Claude Desktop Integration**

**Edit:** `~/.config/claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "seraph": {
      "command": "uv",
      "args": [
        "--directory",
        "/home/malu/.projects/seraph-mcp",
        "run",
        "seraph-mcp"
      ],
      "env": {}
    }
  }
}
```

**Then:** Restart Claude Desktop completely (quit and reopen)

---

## 🧪 Test Compression

**In Claude Desktop, send this prompt:**

```
Use the optimize_context tool to compress this text with hybrid method:

"Artificial intelligence has revolutionized many industries. Machine learning
algorithms can now process vast amounts of data with remarkable speed and accuracy.
Deep learning networks, particularly those using transformer architectures, have
achieved unprecedented results in natural language processing tasks."

Use quality threshold 0.9
```

---

## 📊 Expected Output

**Claude Response:**
```
Compressed text: "AI revolutionized industries. ML algorithms process data fast,
accurately. Deep learning (transformers) achieved unprecedented NLP results."

Metadata:
- Original tokens: 243
- Compressed tokens: 134
- Savings: 109 tokens (45%)
- Quality: 0.92
- Method: hybrid
- Time: 78ms
```

**Server Logs:**
```
INFO - [COMPRESSION] Before: 243 tokens | After: 134 tokens | Saved: 109 tokens (45%) | Method: hybrid | Quality: 0.92 | Time: 78ms
```

---

## ✅ Verification

- [ ] Server starts without errors
- [ ] Claude Desktop shows "Seraph MCP" with 22+ tools
- [ ] `optimize_context` returns compressed text
- [ ] Logs show `[COMPRESSION]` entries
- [ ] `check_budget` shows current spending

---

## 🔧 Troubleshooting

| Issue | Fix |
|-------|-----|
| Server not in Claude | Restart Claude Desktop (quit completely) |
| No logs visible | Set `LOG_LEVEL=DEBUG` in `.env` |
| Tool calls fail | Check API key in `.env`: `grep OPENAI_COMPATIBLE .env` |
| No compression | Verify provider configured: `grep API_KEY .env` |

---

## 📚 Full Documentation

- **Testing Guide:** `docs/TESTING_GUIDE.md`
- **Architecture:** `docs/SDD.md`
- **Environment:** `.env.example`
- **All Tests:** `uv run pytest -v`

---

**🎯 Goal:** See `[COMPRESSION]` logs with 40-60% token savings and >90% quality scores.
