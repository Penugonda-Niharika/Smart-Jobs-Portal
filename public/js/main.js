/**
 * Smart Job Portal — Client-side Interactivity
 * Handles bookmarking via localStorage + session API, mobile navigation,
 * file upload preview, and accordions.
 */

document.addEventListener("DOMContentLoaded", () => {
  initMobileNav();
  initSalaryValidation();
  initSavedJobs();
  initFileUploadDropzone();
  initInterviewAccordions();
});

/**
 * 1. Mobile Menu Toggle
 */
function initMobileNav() {
  const toggleBtn = document.getElementById("mobileToggle");
  const navMenu = document.getElementById("navMenu");

  if (toggleBtn && navMenu) {
    toggleBtn.addEventListener("click", () => {
      navMenu.classList.toggle("open");
    });
  }
}

/**
 * Prevent salary ranges that cannot be submitted and explain the rule inline.
 */
function initSalaryValidation() {
  document.querySelectorAll("form[data-salary-validation]").forEach(form => {
    const minimum = form.querySelector('[name="salary_min"]');
    const maximum = form.querySelector('[name="salary_max"]');
    if (!minimum || !maximum) return;

    const validateRange = () => {
      const minValue = minimum.value === "" ? null : Number(minimum.value);
      const maxValue = maximum.value === "" ? null : Number(maximum.value);
      maximum.setCustomValidity(
        minValue !== null && maxValue !== null && minValue > maxValue
          ? "Maximum salary must be greater than or equal to minimum salary."
          : ""
      );
    };

    minimum.addEventListener("input", validateRange);
    maximum.addEventListener("input", validateRange);
    form.addEventListener("submit", validateRange);
  });
}

/**
 * 2. Saved Jobs (localStorage + Session Sync)
 */
function initSavedJobs() {
  const saveButtons = document.querySelectorAll(".save-job-btn");
  const LOCAL_STORAGE_KEY = "smart_job_portal_saved_ids";

  // Load saved IDs from browser localStorage
  let savedIds = [];
  try {
    const raw = localStorage.getItem(LOCAL_STORAGE_KEY);
    savedIds = raw ? JSON.parse(raw) : [];
  } catch (e) {
    savedIds = [];
  }

  // Update initial active status on buttons
  saveButtons.forEach(btn => {
    const jobId = btn.getAttribute("data-job-id");
    if (savedIds.includes(jobId)) {
      btn.classList.add("active");
      const label = btn.querySelector(".save-label");
      if (label) label.textContent = "Saved";
    }

    btn.addEventListener("click", async (e) => {
      e.preventDefault();
      e.stopPropagation();

      const id = btn.getAttribute("data-job-id");
      if (!id) return;

      const isSaved = btn.classList.toggle("active");
      const label = btn.querySelector(".save-label");
      if (label) label.textContent = isSaved ? "Saved" : "Save";

      // Sync localStorage
      if (isSaved) {
        if (!savedIds.includes(id)) savedIds.push(id);
      } else {
        savedIds = savedIds.filter(item => item !== id);
      }
      localStorage.setItem(LOCAL_STORAGE_KEY, JSON.stringify(savedIds));

      // Sync backend session API
      try {
        const csrfTokenMeta = document.querySelector('meta[name="csrf-token"]');
        const token = csrfTokenMeta ? csrfTokenMeta.getAttribute("content") : "";

        await fetch("/api/toggle-save", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            "X-CSRFToken": token
          },
          body: JSON.stringify({ job_id: id })
        });
      } catch (err) {
        console.warn("Session save sync error:", err);
      }
    });
  });
}

/**
 * 3. File Upload Drag and Drop with Validation
 */
function initFileUploadDropzone() {
  const dropzone = document.getElementById("resumeDropzone");
  const fileInput = document.getElementById("resumeFileInput");
  const fileNameDisplay = document.getElementById("fileNameDisplay");

  if (!dropzone || !fileInput) return;

  dropzone.addEventListener("click", () => fileInput.click());

  dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropzone.classList.add("drag-active");
  });

  ["dragleave", "dragend"].forEach(type => {
    dropzone.addEventListener(type, () => dropzone.classList.remove("drag-active"));
  });

  dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.classList.remove("drag-active");
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      fileInput.files = e.dataTransfer.files;
      handleFileSelected(e.dataTransfer.files[0]);
    }
  });

  fileInput.addEventListener("change", (e) => {
    if (e.target.files && e.target.files.length > 0) {
      handleFileSelected(e.target.files[0]);
    }
  });

  function handleFileSelected(file) {
    if (!file) return;
    const cachedTextEl = document.getElementById("cachedResumeText");
    const cachedFilenameEl = document.getElementById("cachedResumeFilename");
    if (cachedTextEl) cachedTextEl.value = "";
    if (cachedFilenameEl) cachedFilenameEl.value = "";

    const name = file.name;
    const sizeMb = (file.size / (1024 * 1024)).toFixed(2);
    const ext = name.split('.').pop().toLowerCase();

    if (!["pdf", "txt"].includes(ext)) {
      alert("Please select a .PDF or .TXT file. Other formats are unsupported.");
      fileInput.value = "";
      if (fileNameDisplay) fileNameDisplay.textContent = "";
      return;
    }

    if (file.size > 4 * 1024 * 1024) {
      alert("File size exceeds 4 MB. Please upload a smaller file.");
      fileInput.value = "";
      if (fileNameDisplay) fileNameDisplay.textContent = "";
      return;
    }

    if (fileNameDisplay) {
      fileNameDisplay.innerHTML = `<strong>Selected file:</strong> ${name} <span class="text-muted">(${sizeMb} MB)</span>`;
    }
  }
}

/**
 * 4. Interview & Roadmap Accordion Expand/Collapse
 */
function initInterviewAccordions() {
  const toggles = document.querySelectorAll(".accordion-toggle");
  toggles.forEach(btn => {
    btn.addEventListener("click", () => {
      const parent = btn.closest(".accordion-item");
      if (parent) {
        parent.classList.toggle("open");
      }
    });
  });
}
