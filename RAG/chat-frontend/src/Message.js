import React from "react";
import ReactMarkdown from "react-markdown";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { vscDarkPlus } from "react-syntax-highlighter/dist/esm/styles/prism"; // Use a lighter, more readable style

function CopyButton({ code }) {
  const [copied, setCopied] = React.useState(false);
  return (
    <button
      className="absolute top-2 right-2 px-2 py-1 text-xs bg-gray-200 text-gray-700 rounded hover:bg-gray-300 transition border border-gray-300"
      onClick={() => {
        navigator.clipboard.writeText(code);
        setCopied(true);
        setTimeout(() => setCopied(false), 1200);
      }}
      title="Copy code"
      style={{ fontSize: "0.85em" }}
    >
      {copied ? "Copied!" : "Copy"}
    </button>
  );
}

export default function Message({ message, isUser }) {
  return isUser ? (
    // User message: right-aligned blue bubble
    <div className="w-full flex justify-end mb-4">
      <div className="max-w-xl px-4 py-3 rounded-2xl bg-blue-500 text-white shadow rounded-br-none">
        <ReactMarkdown
          children={message}
          components={{
            code({ node, inline, className, children, ...props }) {
              const match = /language-(\w+)/.exec(className || "");
              return !inline ? (
                <div className="relative my-4">
                  <SyntaxHighlighter
                    style={vscDarkPlus}
                    language={match ? match[1] : "python"}
                    PreTag="div"
                    customStyle={{
                      borderRadius: "0.75rem",
                      background: "#f5f5f5",
                      color: "#222",
                      padding: "1.2em",
                      fontSize: "1em",
                      border: "1px solid #e5e7eb",
                    }}
                    {...props}
                  >
                    {String(children).replace(/\n$/, "")}
                  </SyntaxHighlighter>
                  <CopyButton code={String(children)} />
                </div>
              ) : (
                <code className="bg-gray-200 text-gray-800 px-1 rounded">{children}</code>
              );
            },
          }}
        />
      </div>
    </div>
  ) : (
    // AI message: left-aligned, markdown with chatgpt-style code blocks
    <div className="w-full flex justify-start mb-4">
      <div className="max-w-2xl px-6 py-2 text-gray-900 bg-transparent">
        <ReactMarkdown
          children={message}
          components={{
            code({ node, inline, className, children, ...props }) {
              const match = /language-(\w+)/.exec(className || "");
              return !inline ? (
                <div className="relative my-4">
                  <SyntaxHighlighter
                    style={vscDarkPlus}
                    language={match ? match[1] : "python"}
                    PreTag="div"
                    customStyle={{
                      borderRadius: "0.75rem",
                      background: "#f5f5f5",
                      color: "#222",
                      padding: "1.2em",
                      fontSize: "1em",
                      border: "1px solid #e5e7eb",
                    }}
                    {...props}
                  >
                    {String(children).replace(/\n$/, "")}
                  </SyntaxHighlighter>
                  <CopyButton code={String(children)} />
                </div>
              ) : (
                <code className="bg-gray-200 text-gray-800 px-1 rounded">{children}</code>
              );
            },
          }}
        />
      </div>
    </div>
  );
}