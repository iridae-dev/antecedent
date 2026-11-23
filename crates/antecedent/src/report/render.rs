//! Markdown and HTML serialization of the report model.

use super::{Block, Report, clean};

/// Text safe inside Markdown: every markup character is backslash-escaped.
pub(super) fn md_escape(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for c in clean(text).chars() {
        if matches!(c, '\\' | '`' | '*' | '_' | '[' | ']' | '<' | '>' | '|' | '~' | '&' | '#') {
            out.push('\\');
        }
        out.push(c);
    }
    out
}

/// Text safe inside HTML text and quoted attribute values.
pub(super) fn html_escape(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for c in clean(text).chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&#39;"),
            other => out.push(other),
        }
    }
    out
}

pub(super) fn slug(title: &str) -> String {
    title
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c.to_ascii_lowercase() } else { '-' })
        .collect()
}

impl Report {
    pub(super) fn markdown(&self) -> String {
        let mut out = String::new();
        out.push_str("# ");
        out.push_str(&md_escape(self.title));
        out.push_str("\n\n");
        for section in &self.sections {
            out.push_str("## ");
            out.push_str(&md_escape(section.title));
            out.push_str("\n\n");
            for block in &section.blocks {
                md_block(&mut out, block);
            }
        }
        out
    }

    pub(super) fn html(&self) -> String {
        let mut out = String::new();
        out.push_str("<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n");
        out.push_str("<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n");
        out.push_str(
            "<meta http-equiv=\"Content-Security-Policy\" content=\"default-src 'none'; \
             style-src 'unsafe-inline'\">\n",
        );
        out.push_str("<title>");
        out.push_str(&html_escape(self.title));
        out.push_str("</title>\n<style>\n");
        out.push_str(HTML_STYLE);
        out.push_str("</style>\n</head>\n<body>\n<main>\n<h1>");
        out.push_str(&html_escape(self.title));
        out.push_str("</h1>\n");
        for section in &self.sections {
            out.push_str("<section id=\"");
            out.push_str(&html_escape(&slug(section.title)));
            out.push_str("\">\n<h2>");
            out.push_str(&html_escape(section.title));
            out.push_str("</h2>\n");
            for block in &section.blocks {
                html_block(&mut out, block);
            }
            out.push_str("</section>\n");
        }
        out.push_str("</main>\n</body>\n</html>\n");
        out
    }
}

const HTML_STYLE: &str = "\
:root { color-scheme: light dark; --fg: #1b1f24; --bg: #ffffff; --rule: #c9d1d9; --muted: #57606a; }
@media (prefers-color-scheme: dark) { :root { --fg: #e6edf3; --bg: #0d1117; --rule: #30363d; --muted: #8b949e; } }
body { margin: 0; background: var(--bg); color: var(--fg); font: 16px/1.5 system-ui, sans-serif; }
main { max-width: 64rem; margin: 0 auto; padding: 1rem; }
h1, h2, h3 { line-height: 1.25; }
h2 { border-bottom: 1px solid var(--rule); padding-bottom: 0.25rem; margin-top: 2rem; }
dl.facts { display: grid; grid-template-columns: max-content 1fr; gap: 0.25rem 1rem; }
dl.facts dt { font-weight: 600; color: var(--muted); }
dl.facts dd { margin: 0; overflow-wrap: anywhere; }
table { border-collapse: collapse; display: block; overflow-x: auto; }
th, td { border: 1px solid var(--rule); padding: 0.25rem 0.5rem; text-align: left; vertical-align: top; }
";

fn md_block(out: &mut String, block: &Block) {
    match block {
        Block::Facts(facts) => {
            for (key, value) in facts {
                out.push_str("- **");
                out.push_str(&md_escape(key));
                out.push_str(":** ");
                out.push_str(&md_escape(value));
                out.push('\n');
            }
            out.push('\n');
        }
        Block::Text(text) => {
            out.push_str(&md_escape(text));
            out.push_str("\n\n");
        }
        Block::Bullets(items) => {
            for item in items {
                out.push_str("- ");
                out.push_str(&md_escape(item));
                out.push('\n');
            }
            out.push('\n');
        }
        Block::Table { head, rows } => {
            out.push('|');
            for cell in *head {
                out.push(' ');
                out.push_str(&md_escape(cell));
                out.push_str(" |");
            }
            out.push_str("\n|");
            for _ in *head {
                out.push_str(" --- |");
            }
            out.push('\n');
            for row in rows {
                out.push('|');
                for cell in row {
                    out.push(' ');
                    out.push_str(&md_escape(cell));
                    out.push_str(" |");
                }
                out.push('\n');
            }
            out.push('\n');
        }
        Block::Subheading(title) => {
            out.push_str("### ");
            out.push_str(&md_escape(title));
            out.push_str("\n\n");
        }
    }
}

fn html_block(out: &mut String, block: &Block) {
    match block {
        Block::Facts(facts) => {
            out.push_str("<dl class=\"facts\">\n");
            for (key, value) in facts {
                out.push_str("<dt>");
                out.push_str(&html_escape(key));
                out.push_str("</dt><dd>");
                out.push_str(&html_escape(value));
                out.push_str("</dd>\n");
            }
            out.push_str("</dl>\n");
        }
        Block::Text(text) => {
            out.push_str("<p>");
            out.push_str(&html_escape(text));
            out.push_str("</p>\n");
        }
        Block::Bullets(items) => {
            out.push_str("<ul>\n");
            for item in items {
                out.push_str("<li>");
                out.push_str(&html_escape(item));
                out.push_str("</li>\n");
            }
            out.push_str("</ul>\n");
        }
        Block::Table { head, rows } => {
            out.push_str("<table>\n<thead><tr>");
            for cell in *head {
                out.push_str("<th>");
                out.push_str(&html_escape(cell));
                out.push_str("</th>");
            }
            out.push_str("</tr></thead>\n<tbody>\n");
            for row in rows {
                out.push_str("<tr>");
                for cell in row {
                    out.push_str("<td>");
                    out.push_str(&html_escape(cell));
                    out.push_str("</td>");
                }
                out.push_str("</tr>\n");
            }
            out.push_str("</tbody>\n</table>\n");
        }
        Block::Subheading(title) => {
            out.push_str("<h3>");
            out.push_str(&html_escape(title));
            out.push_str("</h3>\n");
        }
    }
}
