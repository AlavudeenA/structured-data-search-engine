// src/extension.ts
import * as vscode from "vscode";
import * as http from "http";
import { URL } from "url";

// Simple helper: read the whole async stream
async function collectStreamText(asyncIterable: AsyncIterable<string>) {
  let out = "";
  for await (const chunk of asyncIterable) out += chunk;
  return out;
}

function startLocalServer(context: vscode.ExtensionContext) {
  // Choose a port or read from config
  const config = vscode.workspace.getConfiguration("vscodeLmSample");
  const port = config.get<number>("localServerPort") ?? 50234;
  const secret = config.get<string>("localServerSecret") ?? "abc123"; // change before sharing

  const server = http.createServer(async (req, res) => {
    try {
      // only accept POST /prompt
      const parsed = new URL(req.url ?? "", `http://localhost`);
      if (req.method !== "POST" || parsed.pathname !== "/prompt") {
        res.writeHead(404, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ error: "not-found" }));
        return;
      }

      // parse JSON body
      const body = await new Promise<string>((resolve, reject) => {
        let buf = "";
        req.on("data", (chunk) => (buf += chunk));
        req.on("end", () => resolve(buf));
        req.on("error", (err) => reject(err));
      });

      let json;
      try {
        json = JSON.parse(body);
      } catch (e) {
        res.writeHead(400, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ error: "invalid-json" }));
        return;
      }

      // simple auth: match secret
      if (!json.secret || json.secret !== secret) {
        res.writeHead(401, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ error: "unauthorized" }));
        return;
      }

      const prompt = String(json.prompt ?? "");
      if (!prompt) {
        res.writeHead(400, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ error: "empty-prompt" }));
        return;
      }

      // Optional system prompt — if caller provides one use it, else fall back to default
      const systemText = json.system
        ? String(json.system)
        : "You are a helpful assistant. Be concise.";

      // Choose the model (user initiated - this is okay because server runs in extension)
      const models = await vscode.lm.selectChatModels({
        vendor: "copilot",
        family: "gpt-4o",
      });
      if (!models || models.length === 0) {
        res.writeHead(503, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ error: "no-model" }));
        return;
      }
      const model = models[0];

      const messages = [
        vscode.LanguageModelChatMessage.User(systemText),
        vscode.LanguageModelChatMessage.User(prompt),
      ];

      // send the request and collect streamed text
      let chatResponse: vscode.LanguageModelChatResponse;
      try {
        chatResponse = await model.sendRequest(
          messages,
          {},
          new vscode.CancellationTokenSource().token,
        );
      } catch (err) {
        const errMsg = err instanceof Error ? err.message : String(err);
        res.writeHead(500, { "Content-Type": "application/json" });
        res.end(
          JSON.stringify({ error: "lm-request-failed", message: errMsg }),
        );
        return;
      }

      // collect full response text
      const text = await collectStreamText(chatResponse.text);

      res.writeHead(200, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ text }));
    } catch (err) {
      console.error("local-server-error", err);
      res.writeHead(500, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ error: "server-error", message: String(err) }));
    }
  });

  server.listen(port, "127.0.0.1", () => {
    console.log(
      `vscode-lm-sample: local server listening on http://127.0.0.1:${port}`,
    );
    vscode.window.showInformationMessage(
      `vscode-lm-sample local API running on port ${port}`,
    );
  });

  context.subscriptions.push({ dispose: () => server.close() });
}

export function activate(context: vscode.ExtensionContext) {
  // register the command declared in package.json
  const disposable = vscode.commands.registerCommand(
    "vscode-lm-sample.askModel",
    async () => {
      try {
        const prompt = await vscode.window.showInputBox({
          prompt: "Ask the language model (brief):",
        });
        if (!prompt) return;

        const models = await vscode.lm.selectChatModels({
          vendor: "copilot",
          family: "gpt-4o",
        });
        if (!models || models.length === 0) {
          vscode.window.showErrorMessage("No language model available");
          return;
        }
        const model = models[0];

        const messages = [
          vscode.LanguageModelChatMessage.User(
            "You are a helpful assistant. Be concise.",
          ),
          vscode.LanguageModelChatMessage.User(prompt),
        ];

        let chatResponse: vscode.LanguageModelChatResponse;
        try {
          chatResponse = await model.sendRequest(
            messages,
            {},
            new vscode.CancellationTokenSource().token,
          );
        } catch (err) {
          vscode.window.showErrorMessage(
            "Model request failed: " + String(err),
          );
          return;
        }

        const text = await collectStreamText(chatResponse.text);
        // show result in an information message (for longer output, consider an output channel)
        const short = text.length > 1000 ? text.slice(0, 1000) + "…" : text;
        vscode.window.showInformationMessage(short);
      } catch (err) {
        vscode.window.showErrorMessage(
          "askModel handler error: " + String(err),
        );
      }
    },
  );

  context.subscriptions.push(disposable);

  // start the local server (only for development/local use)
  startLocalServer(context);
}

export function deactivate() {
  // server closed via subscriptions dispose
}
