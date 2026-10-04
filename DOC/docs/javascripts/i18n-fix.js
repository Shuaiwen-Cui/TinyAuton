// Fix the language switcher when the site is served from a sub-path.
// The logo points to the current language home, which can end in /zh/.
// Resolve the deployment root first and avoid duplicating an existing prefix.
document$.subscribe(function () {
  var logo = document.querySelector(".md-header__button.md-logo");
  if (!logo) return;
  var base = new URL(logo.href);
  base.pathname = base.pathname.replace(/\/zh\/?$/, "/");
  if (!base.pathname.endsWith("/")) base.pathname += "/";
  document.querySelectorAll(".md-select__link").forEach(function (a) {
    var h = a.getAttribute("href");
    if (h && h.charAt(0) === "/") {
      if (base.pathname !== "/" && h.indexOf(base.pathname) === 0) return;
      a.setAttribute("href", new URL(h.slice(1), base).href);
    }
  });
});
