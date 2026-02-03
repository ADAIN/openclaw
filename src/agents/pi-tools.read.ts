import type { AgentToolResult } from "@mariozechner/pi-agent-core";
import { createEditTool, createReadTool, createWriteTool } from "@mariozechner/pi-coding-agent";
import ignore from "ignore";
import fs from "node:fs/promises";
import path from "node:path";
import type { AnyAgentTool } from "./pi-tools.types.js";
import { detectMime } from "../media/mime.js";
import { assertSandboxPath } from "./sandbox-paths.js";
import { sanitizeToolResultImages } from "./tool-images.js";

// NOTE(steipete): Upstream read now does file-magic MIME detection; we keep the wrapper
// to normalize payloads and sanitize oversized images before they hit providers.
type ToolContentBlock = AgentToolResult<unknown>["content"][number];
type ImageContentBlock = Extract<ToolContentBlock, { type: "image" }>;
type TextContentBlock = Extract<ToolContentBlock, { type: "text" }>;

async function sniffMimeFromBase64(base64: string): Promise<string | undefined> {
  const trimmed = base64.trim();
  if (!trimmed) {
    return undefined;
  }

  const take = Math.min(256, trimmed.length);
  const sliceLen = take - (take % 4);
  if (sliceLen < 8) {
    return undefined;
  }

  try {
    const head = Buffer.from(trimmed.slice(0, sliceLen), "base64");
    return await detectMime({ buffer: head });
  } catch {
    return undefined;
  }
}

function rewriteReadImageHeader(text: string, mimeType: string): string {
  // pi-coding-agent uses: "Read image file [image/png]"
  if (text.startsWith("Read image file [") && text.endsWith("]")) {
    return `Read image file [${mimeType}]`;
  }
  return text;
}

async function normalizeReadImageResult(
  result: AgentToolResult<unknown>,
  filePath: string,
): Promise<AgentToolResult<unknown>> {
  const content = Array.isArray(result.content) ? result.content : [];

  const image = content.find(
    (b): b is ImageContentBlock =>
      !!b &&
      typeof b === "object" &&
      (b as { type?: unknown }).type === "image" &&
      typeof (b as { data?: unknown }).data === "string" &&
      typeof (b as { mimeType?: unknown }).mimeType === "string",
  );
  if (!image) {
    return result;
  }

  if (!image.data.trim()) {
    throw new Error(`read: image payload is empty (${filePath})`);
  }

  const sniffed = await sniffMimeFromBase64(image.data);
  if (!sniffed) {
    return result;
  }

  if (!sniffed.startsWith("image/")) {
    throw new Error(
      `read: file looks like ${sniffed} but was treated as ${image.mimeType} (${filePath})`,
    );
  }

  if (sniffed === image.mimeType) {
    return result;
  }

  const nextContent = content.map((block) => {
    if (block && typeof block === "object" && (block as { type?: unknown }).type === "image") {
      const b = block as ImageContentBlock & { mimeType: string };
      return { ...b, mimeType: sniffed } satisfies ImageContentBlock;
    }
    if (
      block &&
      typeof block === "object" &&
      (block as { type?: unknown }).type === "text" &&
      typeof (block as { text?: unknown }).text === "string"
    ) {
      const b = block as TextContentBlock & { text: string };
      return {
        ...b,
        text: rewriteReadImageHeader(b.text, sniffed),
      } satisfies TextContentBlock;
    }
    return block;
  });

  return { ...result, content: nextContent };
}

type RequiredParamGroup = {
  keys: readonly string[];
  allowEmpty?: boolean;
  label?: string;
};

export const CLAUDE_PARAM_GROUPS = {
  read: [{ keys: ["path", "file_path"], label: "path (path or file_path)" }],
  write: [{ keys: ["path", "file_path"], label: "path (path or file_path)" }],
  edit: [
    { keys: ["path", "file_path"], label: "path (path or file_path)" },
    {
      keys: ["oldText", "old_string"],
      label: "oldText (oldText or old_string)",
    },
    {
      keys: ["newText", "new_string"],
      label: "newText (newText or new_string)",
    },
  ],
} as const;

// Normalize tool parameters from Claude Code conventions to pi-coding-agent conventions.
// Claude Code uses file_path/old_string/new_string while pi-coding-agent uses path/oldText/newText.
// This prevents models trained on Claude Code from getting stuck in tool-call loops.
export function normalizeToolParams(params: unknown): Record<string, unknown> | undefined {
  if (!params || typeof params !== "object") {
    return undefined;
  }
  const record = params as Record<string, unknown>;
  const normalized = { ...record };
  // file_path → path (read, write, edit)
  if ("file_path" in normalized && !("path" in normalized)) {
    normalized.path = normalized.file_path;
    delete normalized.file_path;
  }
  // old_string → oldText (edit)
  if ("old_string" in normalized && !("oldText" in normalized)) {
    normalized.oldText = normalized.old_string;
    delete normalized.old_string;
  }
  // new_string → newText (edit)
  if ("new_string" in normalized && !("newText" in normalized)) {
    normalized.newText = normalized.new_string;
    delete normalized.new_string;
  }
  return normalized;
}

export function patchToolSchemaForClaudeCompatibility(tool: AnyAgentTool): AnyAgentTool {
  const schema =
    tool.parameters && typeof tool.parameters === "object"
      ? (tool.parameters as Record<string, unknown>)
      : undefined;

  if (!schema || !schema.properties || typeof schema.properties !== "object") {
    return tool;
  }

  const properties = { ...(schema.properties as Record<string, unknown>) };
  const required = Array.isArray(schema.required)
    ? schema.required.filter((key): key is string => typeof key === "string")
    : [];
  let changed = false;

  const aliasPairs: Array<{ original: string; alias: string }> = [
    { original: "path", alias: "file_path" },
    { original: "oldText", alias: "old_string" },
    { original: "newText", alias: "new_string" },
  ];

  for (const { original, alias } of aliasPairs) {
    if (!(original in properties)) {
      continue;
    }
    if (!(alias in properties)) {
      properties[alias] = properties[original];
      changed = true;
    }
    const idx = required.indexOf(original);
    if (idx !== -1) {
      required.splice(idx, 1);
      changed = true;
    }
  }

  if (!changed) {
    return tool;
  }

  return {
    ...tool,
    parameters: {
      ...schema,
      properties,
      required,
    },
  };
}

export function assertRequiredParams(
  record: Record<string, unknown> | undefined,
  groups: readonly RequiredParamGroup[],
  toolName: string,
): void {
  if (!record || typeof record !== "object") {
    throw new Error(`Missing parameters for ${toolName}`);
  }

  for (const group of groups) {
    const satisfied = group.keys.some((key) => {
      if (!(key in record)) {
        return false;
      }
      const value = record[key];
      if (typeof value !== "string") {
        return false;
      }
      if (group.allowEmpty) {
        return true;
      }
      return value.trim().length > 0;
    });

    if (!satisfied) {
      const label = group.label ?? group.keys.join(" or ");
      throw new Error(`Missing required parameter: ${label}`);
    }
  }
}

// Generic wrapper to normalize parameters for any tool
export function wrapToolParamNormalization(
  tool: AnyAgentTool,
  requiredParamGroups?: readonly RequiredParamGroup[],
): AnyAgentTool {
  const patched = patchToolSchemaForClaudeCompatibility(tool);
  return {
    ...patched,
    execute: async (toolCallId, params, signal, onUpdate) => {
      const normalized = normalizeToolParams(params);
      const record =
        normalized ??
        (params && typeof params === "object" ? (params as Record<string, unknown>) : undefined);
      if (requiredParamGroups?.length) {
        assertRequiredParams(record, requiredParamGroups, tool.name);
      }
      return tool.execute(toolCallId, normalized ?? params, signal, onUpdate);
    },
  };
}

function wrapSandboxPathGuard(tool: AnyAgentTool, root: string): AnyAgentTool {
  return {
    ...tool,
    execute: async (toolCallId, args, signal, onUpdate) => {
      const normalized = normalizeToolParams(args);
      const record =
        normalized ??
        (args && typeof args === "object" ? (args as Record<string, unknown>) : undefined);
      const filePath = record?.path;
      if (typeof filePath === "string" && filePath.trim()) {
        const { resolved } = await assertSandboxPath({ filePath, cwd: root, root });
        await checkIgnorePolicy(resolved, root);
      }
      return tool.execute(toolCallId, normalized ?? args, signal, onUpdate);
    },
  };
}

// Checks for .ignore files in all parent directories up to root.
// Collects rules from all .ignore files and checks if the target path is ignored.
// Emulates hierarchical behavior where rules are accumulated.
async function checkIgnorePolicy(absoluteFilePath: string, root: string): Promise<void> {
  const rootResolved = path.resolve(root);
  // Calculate relative path from root to the target file
  const relativeToRoot = path.relative(rootResolved, absoluteFilePath);

  if (relativeToRoot.startsWith("..") || path.isAbsolute(relativeToRoot)) {
    // Should have been caught by assertSandboxPath, but just in case
    return;
  }

  const ig = ignore();

  // Traverse from root down to the file's directory
  // For each directory, look for .ignore and add its rules
  // The trick is to scope the rules to that directory.
  // ignore() matches against the path string provided to ignores().
  // If we add rules like "foo.txt" from "sub/.ignore", and we test "sub/foo.txt",
  // we need the rule to be effectively "sub/foo.txt".
  // However, `ignore` library supports relative paths if we treat the whole thing as one repo.
  // But standard .gitignore behavior implies rules in subdirs are relative to that subdir.
  // The 'ignore' package documentation says:
  // "There is no current plan to support multiple .gitignore files in nested directories."
  // So we have to simulate it manually or simplistically.

  // STRATEGY:
  // We collect all .ignore files from root to target dir.
  // For each .ignore file found at `dir`, we read its rules.
  // We transform each rule: if rule is `pattern`, we prefix it with `path.relative(root, dir)`.
  // e.g. root/.ignore: "*.log" -> "*.log"
  //      root/sub/.ignore: "*.secret" -> "sub/*.secret"
  //      root/sub/.ignore: "!foo.secret" -> "!sub/foo.secret"
  // This allows us to feed a single global path (relative to root) to `ig.ignores()`.

  const targetDir = path.dirname(absoluteFilePath);
  let currentDir = rootResolved;

  // We need to walk step by step from root to targetDir
  // Construct path segments
  const segments = relativeToRoot.split(path.sep);
  // The file itself is the last segment, we want to check directories up to its parent
  // actually, we should check .ignore in the file's directory too (if it exists)
  // But `relativeToRoot` includes the filename. `path.dirname` handles it.

  // List of directories to check: root, root/sub1, root/sub1/sub2, ...
  const dirsToCheck: string[] = [rootResolved];
  let accumPath = rootResolved;

  // Remove filename from segments to get directories
  const dirSegments = segments.slice(0, -1);

  for (const segment of dirSegments) {
    accumPath = path.join(accumPath, segment);
    dirsToCheck.push(accumPath);
  }

  for (const dir of dirsToCheck) {
    const ignorePath = path.join(dir, ".ignore");
    try {
      const content = await fs.readFile(ignorePath, "utf8");

      // Calculate prefix for this directory relative to root
      // e.g. "" for root, "sub/" for sub
      let prefix = path.relative(rootResolved, dir);
      if (prefix && !prefix.endsWith("/")) {
        prefix += "/";
      }
      if (prefix === "") prefix = ""; // Ensure empty string for root

      const rules = content.split(/\r?\n/);
      const scopedRules: string[] = [];

      for (let rule of rules) {
        rule = rule.trim();
        if (!rule || rule.startsWith("#")) continue;

        // Handle negation
        const isNegative = rule.startsWith("!");
        if (isNegative) {
          rule = rule.slice(1);
        }

        // Handle root-anchored rules in gitignore (starting with /)
        // In a subdir .gitignore, /foo means subdir/foo.
        if (rule.startsWith("/")) {
          rule = rule.slice(1);
        }

        // Combine prefix and rule
        let scopedRule = prefix + rule;

        if (isNegative) {
          scopedRule = "!" + scopedRule;
        }

        scopedRules.push(scopedRule);
      }

      ig.add(scopedRules);
    } catch (err: any) {
      if (err.code !== "ENOENT") {
        // Log error but allow access? Or strict deny?
        // Let's allow access if we can't read .ignore (e.g. permission error might be weird)
        // but safest is to ignore the error (treat as no .ignore)
      }
    }
  }

  if (ig.ignores(relativeToRoot)) {
    throw new Error(`Access denied: Path ${relativeToRoot} is ignored by .ignore policy.`);
  }
}

export function createSandboxedReadTool(root: string) {
  const base = createReadTool(root) as unknown as AnyAgentTool;
  return wrapSandboxPathGuard(createOpenClawReadTool(base), root);
}

export function createSandboxedWriteTool(root: string) {
  const base = createWriteTool(root) as unknown as AnyAgentTool;
  return wrapSandboxPathGuard(wrapToolParamNormalization(base, CLAUDE_PARAM_GROUPS.write), root);
}

export function createSandboxedEditTool(root: string) {
  const base = createEditTool(root) as unknown as AnyAgentTool;
  return wrapSandboxPathGuard(wrapToolParamNormalization(base, CLAUDE_PARAM_GROUPS.edit), root);
}

export function createOpenClawReadTool(base: AnyAgentTool): AnyAgentTool {
  const patched = patchToolSchemaForClaudeCompatibility(base);
  return {
    ...patched,
    execute: async (toolCallId, params, signal) => {
      const normalized = normalizeToolParams(params);
      const record =
        normalized ??
        (params && typeof params === "object" ? (params as Record<string, unknown>) : undefined);
      assertRequiredParams(record, CLAUDE_PARAM_GROUPS.read, base.name);
      const result = await base.execute(toolCallId, normalized ?? params, signal);
      const filePath = typeof record?.path === "string" ? String(record.path) : "<unknown>";
      const normalizedResult = await normalizeReadImageResult(result, filePath);
      return sanitizeToolResultImages(normalizedResult, `read:${filePath}`);
    },
  };
}
