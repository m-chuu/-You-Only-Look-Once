#!/usr/bin/python3
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import serial
import serial.tools.list_ports
import threading
import time
import os
import zlib
from datetime import datetime
import numpy as np
from PIL import Image, ImageTk

class AIBoundingBoxGUI:
    def __init__(self):
        # 1. Create the main window FIRST
        self.window = tk.Tk()
        self.window.title("Logikhaus SoC Control System v2.1")
        self.window.geometry("950x850") 

        # 2. Initialize Variables
        self.serial_port = None
        self.is_connected = False
        self.is_transferring = False
        self.bbox_history = []
        self.last_mem_check_index = None
        
        self.var_latest_event = tk.StringVar(value="System Ready")
        
        # Shared Variables
        self.vpu_mode = tk.IntVar(value=0) 
        self.vpu_level = tk.DoubleVar(value=50.0)
        
        self.ai_addresses = {
            "weight_file.bin": "01000000",
            "padding_file.bin": "04800000",
            "img0_file.bin": "03000000"
        }

        self.status_vars = {
            "weight": tk.StringVar(value="Weight: ✗"),
            "padding": tk.StringVar(value="Padding: ✗"),
            "image": tk.StringVar(value="Image: ✗")
        }

        # 3. Setup UI
        self.setup_ui()
        
    def setup_ui(self):
        self.window.columnconfigure(0, weight=1)
        self.window.rowconfigure(0, weight=0) # Title
        self.window.rowconfigure(1, weight=1) # Notebook
        
        # Title Header
        header_frame = tk.Frame(self.window, bg="#333")
        header_frame.grid(row=0, column=0, sticky="ew")
        tk.Label(header_frame, text="Logikhaus SoC Control System", 
                 font=("Arial", 14, "bold"), fg="white", bg="#333", pady=8).pack()

        # Notebook
        notebook = ttk.Notebook(self.window)
        notebook.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        
        # Create Tabs
        self.setup_dashboard_tab(notebook)
        self.setup_ai_control_tab(notebook) 
        self.setup_vpu_tab(notebook)        
        self.setup_console_tab(notebook)
        
        # Status Bar
        self.window.rowconfigure(2, weight=0)
        self.status_bar = tk.Label(self.window, text="Ready", bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.grid(row=2, column=0, sticky="ew")
        
    def setup_dashboard_tab(self, notebook):
        tab = tk.Frame(notebook)
        notebook.add(tab, text="Dashboard")
        
        tab.columnconfigure(0, weight=1)
        tab.columnconfigure(1, weight=1)
        tab.rowconfigure(2, weight=1) 

        # --- Section 1: Connection ---
        conn_frame = tk.LabelFrame(tab, text="Connection", padx=5, pady=5, font=("Arial", 10, "bold"))
        conn_frame.grid(row=0, column=0, columnspan=2, sticky="ew", padx=10, pady=5)
        conn_frame.columnconfigure(1, weight=1)

        self.update_ports_list()
        self.port_var = tk.StringVar(value=self.ports[0] if self.ports else "")
        ttk.Combobox(conn_frame, textvariable=self.port_var, values=self.ports, width=20).grid(row=0, column=0, padx=5)
        
        self.baud_var = tk.StringVar(value="115200")
        ttk.Combobox(conn_frame, textvariable=self.baud_var, values=["115200", "230400"], width=10).grid(row=0, column=1, padx=5, sticky="w")
        
        tk.Button(conn_frame, text="Refresh", command=self.refresh_ports).grid(row=0, column=2, padx=2)
        self.connect_btn = tk.Button(conn_frame, text="Connect", command=self.toggle_connection, bg="lightgreen", width=12)
        self.connect_btn.grid(row=0, column=3, padx=5)

        # --- Section 2: System Status Panel ---
        stat_frame = tk.LabelFrame(tab, text="AI Files Status", padx=10, pady=10, font=("Arial", 10, "bold"))
        stat_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        
        tk.Label(stat_frame, textvariable=self.status_vars["weight"], font=("Arial", 11), fg="darkblue").pack(anchor="w", pady=2)
        tk.Label(stat_frame, textvariable=self.status_vars["padding"], font=("Arial", 11), fg="darkblue").pack(anchor="w", pady=2)
        tk.Label(stat_frame, textvariable=self.status_vars["image"], font=("Arial", 11), fg="darkblue").pack(anchor="w", pady=2)
        
        tk.Button(stat_frame, text="Check", command=lambda: self.send_serial("S\n"), bg="#ddd").pack(fill=tk.X, pady=10)

        # --- Section 3: DDR Memory Monitor ---
        self.setup_memory_monitor(tab)

        # --- Section 4: Quick Controls ---
        ctrl_panel = tk.Frame(tab)
        ctrl_panel.grid(row=2, column=0, columnspan=2, sticky="nsew", padx=10, pady=5)
        ctrl_panel.columnconfigure(0, weight=1)
        ctrl_panel.columnconfigure(1, weight=1)

        ai_frame = tk.LabelFrame(ctrl_panel, text="AI Quick Actions", padx=5, pady=5, fg="maroon")
        ai_frame.grid(row=0, column=0, sticky="nsew", padx=5)
        
        tk.Button(ai_frame, text="🤖 Auto-Detect (AI)", command=lambda: self.send_serial("W4003000000000001;\n"), 
                  bg="lightyellow", height=2).pack(fill=tk.X, pady=5)
        tk.Button(ai_frame, text="❌ Clear BBox", command=self.send_clear_sequence, 
                  bg="lightcoral", height=2).pack(fill=tk.X, pady=5)

        vpu_frame = tk.LabelFrame(ctrl_panel, text="VPU Quick Control", padx=5, pady=5, fg="maroon")
        vpu_frame.grid(row=0, column=1, sticky="nsew", padx=5)
        
        tk.Radiobutton(vpu_frame, text="Bypass", variable=self.vpu_mode, value=0, command=self.on_vpu_mode_change).pack(anchor="w")
        tk.Radiobutton(vpu_frame, text="Edge Detect", variable=self.vpu_mode, value=1, command=self.on_vpu_mode_change).pack(anchor="w")
        tk.Radiobutton(vpu_frame, text="Sharpen", variable=self.vpu_mode, value=2, command=self.on_vpu_mode_change).pack(anchor="w")
        
        tk.Scale(vpu_frame, from_=0, to=100, orient=tk.HORIZONTAL, variable=self.vpu_level, 
                 command=self.on_vpu_level_change).pack(fill=tk.X, pady=5)

        # --- Section 5: LATEST EVENTS ---
        event_frame = tk.LabelFrame(tab, text="Latest System Event", padx=5, pady=5, bg="#f0f0f0")
        event_frame.grid(row=2, column=0, columnspan=2, sticky="ew", padx=10, pady=10)
        tk.Label(event_frame, textvariable=self.var_latest_event, font=("Consolas", 10), 
                 fg="#333", bg="#f0f0f0", anchor="w").pack(fill=tk.X)

    def setup_memory_monitor(self, parent):
        mem_frame = tk.LabelFrame(parent, text="DDR Memory Monitor (First 4 Bytes)", padx=5, pady=5, font=("Arial", 10, "bold"))
        mem_frame.grid(row=1, column=1, sticky="nsew", padx=5, pady=5)
        
        headers = ["Region", "Address", "Value (Hex)", "Action"]
        for i, h in enumerate(headers):
            tk.Label(mem_frame, text=h, font=("Arial", 9, "bold")).grid(row=0, column=i, padx=5, pady=2)

        self.mem_monitors = [
            {"label": "Weights", "addr": "01000000", "var": tk.StringVar(value="--")},
            {"label": "Padding", "addr": "04800000", "var": tk.StringVar(value="--")},
            {"label": "Image",   "addr": "03000000", "var": tk.StringVar(value="--")},
            {"label": "BBox Tbl","addr": "00002000", "var": tk.StringVar(value="--")}, 
        ]

        for i, item in enumerate(self.mem_monitors):
            row = i + 1
            tk.Label(mem_frame, text=item["label"]).grid(row=row, column=0, sticky="w", padx=5)
            tk.Label(mem_frame, text=f"0x{item['addr']}", fg="gray").grid(row=row, column=1, padx=5)
            tk.Label(mem_frame, textvariable=item["var"], fg="blue", font=("Consolas", 10)).grid(row=row, column=2, padx=5)
            
            cmd = f"R{item['addr']}00000004;\n"
            btn = tk.Button(mem_frame, text="Check", command=lambda c=cmd, idx=i: self.check_memory(c, idx), height=1, font=("Arial", 8))
            btn.grid(row=row, column=3, padx=5, pady=2)

    def check_memory(self, cmd, index):
        self.last_mem_check_index = index 
        self.send_serial(cmd)

    def send_clear_sequence(self):
        self.send_serial("W4003000000000000;\n")
        self.window.after(50, lambda: self.send_serial("C;\n"))

    def setup_ai_control_tab(self, notebook):
        tab = tk.Frame(notebook)
        notebook.add(tab, text="AI Control")
        
        tab.rowconfigure(0, weight=0)
        tab.rowconfigure(1, weight=1)
        tab.columnconfigure(0, weight=1)

        top_frame = tk.Frame(tab)
        top_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        top_frame.columnconfigure(0, weight=1)
        top_frame.columnconfigure(1, weight=1)

        # ==========================================
        # LEFT COLUMN: Models and Images
        # ==========================================
        left_col = tk.Frame(top_frame)
        left_col.grid(row=0, column=0, sticky="nsew", padx=5)

        # --- 1. AI Model Storage ---
        model_frame = tk.LabelFrame(left_col, text="1. AI Model Storage (SD Card)", padx=5, pady=5)
        model_frame.pack(fill=tk.X, pady=(0, 5))

        tk.Label(model_frame, text="File:").grid(row=0, column=0, sticky="w")
        self.model_file_var = tk.StringVar(value="weight_file.bin")
        ttk.Combobox(model_frame, textvariable=self.model_file_var, values=["weight_file.bin", "padding_file.bin"], state="readonly", width=15).grid(row=0, column=1, sticky="ew", padx=5)

        self.model_path_var = tk.StringVar(value="No file")
        tk.Entry(model_frame, textvariable=self.model_path_var, state="readonly", width=20).grid(row=1, column=0, columnspan=2, sticky="ew", pady=5)
        tk.Button(model_frame, text="Browse", command=lambda: self.browse_file(self.model_path_var, self.model_size_lbl, "model")).grid(row=1, column=2, padx=2)

        self.model_size_lbl = tk.Label(model_frame, text="Size: 0 KB", font=("Consolas", 8))
        self.model_size_lbl.grid(row=2, column=0, columnspan=2, sticky="w")

        m_btn_frame = tk.Frame(model_frame)
        m_btn_frame.grid(row=3, column=0, columnspan=3, sticky="ew", pady=5)
        m_btn_frame.columnconfigure(0, weight=1)
        m_btn_frame.columnconfigure(1, weight=1)

        self.upload_model_btn = tk.Button(m_btn_frame, text="UPLOAD TO SD", command=self.upload_model, bg="lightblue", height=2)
        self.upload_model_btn.grid(row=0, column=0, sticky="ew", padx=(0, 2))

        self.load_sd_btn = tk.Button(m_btn_frame, text="LOAD FROM SD\nTO RAM", command=self.load_from_sd, bg="#ffcc99", height=2)
        self.load_sd_btn.grid(row=0, column=1, sticky="ew", padx=(2, 0))

        self.model_prog = ttk.Progressbar(model_frame, length=100, mode='determinate')
        self.model_prog.grid(row=4, column=0, columnspan=2, sticky="ew", pady=2)
        self.model_pct = tk.Label(model_frame, text="0%", font=("Arial", 9))
        self.model_pct.grid(row=4, column=2, sticky="w", padx=2)

        # --- 2. Image Input ---
        img_frame = tk.LabelFrame(left_col, text="2. Image Input (DDR RAM)", padx=5, pady=5)
        img_frame.pack(fill=tk.X, pady=(5, 0))

        tk.Label(img_frame, text="Name:").grid(row=0, column=0, sticky="w")
        self.img_name_entry = tk.Entry(img_frame, width=15)
        self.img_name_entry.insert(0, "img0_file.bin")
        self.img_name_entry.grid(row=0, column=1, sticky="ew", padx=5)

        self.img_path_var = tk.StringVar(value="No file")
        tk.Entry(img_frame, textvariable=self.img_path_var, state="readonly", width=20).grid(row=1, column=0, columnspan=2, sticky="ew", pady=5)
        tk.Button(img_frame, text="Browse", command=lambda: self.browse_file(self.img_path_var, self.img_size_lbl, "image")).grid(row=1, column=2, padx=2)

        self.img_size_lbl = tk.Label(img_frame, text="Size: 0 KB", font=("Consolas", 8))
        self.img_size_lbl.grid(row=2, column=0, columnspan=2, sticky="w")

        self.upload_img_btn = tk.Button(img_frame, text="UPLOAD IMAGE TO RAM", command=self.upload_image, bg="lightgreen", height=2)
        self.upload_img_btn.grid(row=3, column=0, columnspan=3, sticky="ew", pady=5)

        self.img_prog = ttk.Progressbar(img_frame, length=100, mode='determinate')
        self.img_prog.grid(row=4, column=0, columnspan=2, sticky="ew", pady=2)
        self.img_pct = tk.Label(img_frame, text="0%", font=("Arial", 9))
        self.img_pct.grid(row=4, column=2, sticky="w", padx=2)

        # ==========================================
        # RIGHT COLUMN: BBox Control
        # ==========================================
        bbox_frame = tk.LabelFrame(top_frame, text="Bounding Box Control", padx=5, pady=5)
        bbox_frame.grid(row=0, column=1, sticky="nsew", padx=5)

        tk.Button(bbox_frame, text="RUN AI DETECTION", command=lambda: self.send_serial("W4003000000000001;\n"), 
                  bg="lightyellow", font=("Arial", 9, "bold")).pack(fill=tk.X, pady=2)
        
        man_grid = tk.Frame(bbox_frame)
        man_grid.pack(fill=tk.X, pady=5)
        tk.Label(man_grid, text="L:").pack(side=tk.LEFT)
        self.entry_l = tk.Entry(man_grid, width=4); self.entry_l.pack(side=tk.LEFT)
        tk.Label(man_grid, text="R:").pack(side=tk.LEFT)
        self.entry_r = tk.Entry(man_grid, width=4); self.entry_r.pack(side=tk.LEFT)
        tk.Label(man_grid, text="T:").pack(side=tk.LEFT)
        self.entry_t = tk.Entry(man_grid, width=4); self.entry_t.pack(side=tk.LEFT)
        tk.Label(man_grid, text="B:").pack(side=tk.LEFT)
        self.entry_b = tk.Entry(man_grid, width=4); self.entry_b.pack(side=tk.LEFT)
        
        tk.Button(bbox_frame, text="Draw Manual Box", command=self.send_custom_bbox).pack(fill=tk.X, pady=2)
        
        btn_row = tk.Frame(bbox_frame)
        btn_row.pack(fill=tk.X)
        tk.Button(btn_row, text="Center", width=6, command=lambda: self.send_manual_cmd(860, 1060, 440, 640)).pack(side=tk.LEFT, padx=1)
        tk.Button(btn_row, text="Corner", width=6, command=lambda: self.send_manual_cmd(100, 300, 100, 300)).pack(side=tk.LEFT, padx=1)
        tk.Button(btn_row, text="Clear", width=6, bg="lightcoral", command=self.send_clear_sequence).pack(side=tk.LEFT, padx=1)
        
        vis_frame = tk.Frame(bbox_frame)
        vis_frame.pack(fill=tk.X, pady=5)
        tk.Label(vis_frame, text="Thickness:").pack(side=tk.LEFT)
        tk.Scale(vis_frame, from_=0, to=15, orient=tk.HORIZONTAL, command=lambda v: self.send_serial(f"T{v};\n", False)).pack(side=tk.LEFT, fill=tk.X, expand=True)

        # ==========================================
        # BOTTOM: History
        # ==========================================
        hist_frame = tk.LabelFrame(tab, text="Detection History", padx=5, pady=5)
        hist_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=5)
        
        list_container = tk.Frame(hist_frame)
        list_container.pack(side=tk.TOP, fill=tk.X, expand=False)

        cols = ("Time", "Source", "Left", "Right", "Top", "Bottom")
        self.bbox_tree = ttk.Treeview(list_container, columns=cols, show="headings", height=12)
        
        for col in cols:
            self.bbox_tree.heading(col, text=col)
            self.bbox_tree.column(col, width=70, anchor="center")
            
        sb = ttk.Scrollbar(list_container, orient=tk.VERTICAL, command=self.bbox_tree.yview)
        self.bbox_tree.configure(yscrollcommand=sb.set)
        
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.bbox_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        tk.Button(hist_frame, text="🗑 Reset History", command=self.clear_bbox_history, 
                  bg="salmon").pack(side=tk.TOP, fill=tk.X, pady=2)

    # ==========================================
    # FILE UPLOAD LOGIC
    # ==========================================
    def browse_file(self, path_var, size_lbl, file_type):
            f = filedialog.askopenfilename(filetypes=[("Binary", "*.bin"), ("All", "*.*")])
            if f:
                if file_type == "model":
                    self.model_full_path = f
                else:
                    self.img_full_path = f
                    # self.render_bin_preview(f) 
                    
                path_var.set(os.path.basename(f))
                size = os.path.getsize(f)
                size_lbl.config(text=f"Size: {size:,} bytes")

    # ---> NEW: Render Binary Preview Function <---
# ---> NEW: Render Binary Preview Function <---
    def render_bin_preview(self, filepath):
        try:
            with open(filepath, 'rb') as f:
                raw_data = f.read()

            width, height = 416, 256
            # UPDATE: 4 channels, 2 bytes per float16
            expected_size = width * height * 4 * 2 

            if len(raw_data) == expected_size:
                # Use the exact same rendering logic as the live preview
                pixel_array = np.frombuffer(raw_data, dtype=np.float16)
                pixel_matrix = pixel_array.reshape((height, width, 4))
                rgb_matrix = pixel_matrix[:, :, :3]
                rgb_matrix = np.clip(rgb_matrix * 255.0, 0, 255).astype(np.uint8)
                
                pil_img = Image.fromarray(rgb_matrix, 'RGB')
                pil_img.thumbnail((1920, 1080)) 
                
                self.preview_photo = ImageTk.PhotoImage(pil_img) 
                self.img_preview_lbl.config(image=self.preview_photo, text="")
            else:
                self.img_preview_lbl.config(image='', text=f"Format Mismatch\nExpected {expected_size} bytes\nGot {len(raw_data)} bytes\n(Ensure image is FP16 Padded)")
                
        except Exception as e:
            self.img_preview_lbl.config(image='', text=f"Preview Error:\n{str(e)}")

    def upload_model(self):
        if not self.is_connected or not hasattr(self, 'model_full_path'): return
        fname = self.model_file_var.get()
        addr = self.ai_addresses.get(fname, "00000000")
        
        threading.Thread(target=self._upload_thread, 
                         args=(addr, fname, self.model_full_path, self.upload_model_btn, self.model_prog, self.model_pct), 
                         daemon=True).start()

    def upload_image(self):
        if not self.is_connected or not hasattr(self, 'img_full_path'): return
        fname = self.img_name_entry.get()
        addr = self.ai_addresses.get("img0_file.bin", "03000000")
        
        threading.Thread(target=self._upload_thread, 
                         args=(addr, fname, self.img_full_path, self.upload_img_btn, self.img_prog, self.img_pct), 
                         daemon=True).start()
    
    def load_from_sd(self):
        if not self.is_connected: return
        # Disable button and set to simple loading text
        self.load_sd_btn.config(state="disabled", text="LOADING...\n(Please Wait)")
        self.send_serial("O\n")
    
    def sd_load_complete(self):
        # Flash the button green to indicate success
        self.load_sd_btn.config(state="normal", text="SUCCESS!\nLOADED TO RAM", bg="lightgreen")
        # Revert button appearance after 4 seconds
        self.window.after(4000, lambda: self.load_sd_btn.config(text="LOAD FROM SD\nTO RAM", bg="#ffcc99"))

    def _upload_thread(self, address, filename, filepath, btn, prog_bar, pct_lbl):
        try:
            self.is_transferring = True
            original_text = btn.cget("text")
            btn.config(state="disabled", text="Uploading...")
            with open(filepath, 'rb') as f:
                data = f.read()
            size_hex = format(len(data)*8, '08X')
            crc = format(zlib.crc32(data) & 0xFFFFFFFF, '08X')
            
            self.send_serial(f"A{address}{filename};")
            time.sleep(0.5)
            self.send_serial(size_hex)
            time.sleep(0.5)
            
            chunk_sz = 4096 
            for i in range(0, len(data), chunk_sz):
                self.serial_port.write(data[i:i+chunk_sz])
                
                # Only update the GUI every few chunks so Tkinter doesn't freeze
                if i % (chunk_sz * 10) == 0:
                    pct = (i / len(data)) * 100
                    prog_bar['value'] = pct
                    self.window.after(0, lambda p=pct: pct_lbl.config(text=f"{int(p)}%"))
                
            time.sleep(0.5)
            self.send_serial(crc)
            
            prog_bar['value'] = 100
            self.window.after(0, lambda: pct_lbl.config(text="100%")) 
            self.log(f"Uploaded {filename}")
            
        except Exception as e:
            self.log(f"Upload error: {e}")
        finally:
            self.is_transferring = False
            btn.config(state="normal", text=original_text)

    # ==========================================
    # VPU CONTROL TAB (UPDATED WITH IMAGE PREVIEW)
    # ==========================================

    def setup_vpu_tab(self, notebook):
        tab = tk.Frame(notebook)
        notebook.add(tab, text="VPU Control")
        tab.columnconfigure(0, weight=1)
        
        # Give the preview frame room to expand
        tab.rowconfigure(2, weight=1)

        # --- Section 1: Mode Selection ---
        mode_frame = tk.LabelFrame(tab, text="Mode Selection", padx=10, pady=10)
        mode_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=5)

        modes = [("Video Bypass", 0), ("Edge Detection", 1), ("Sharpening", 2)]
        for text, val in modes:
            tk.Radiobutton(mode_frame, text=text, variable=self.vpu_mode, value=val, 
                           command=self.on_vpu_mode_change).pack(side=tk.LEFT, expand=True)

       

        # --- Section 3: Effect Level Frame ---
        self.level_frame = tk.LabelFrame(tab, text="Effect Level (0-100%)", padx=10, pady=10)
        self.level_frame.grid(row=1, column=0, sticky="ew", padx=5, pady=5)
        
        self.vpu_level_scale = tk.Scale(self.level_frame, from_=0, to=100, orient=tk.HORIZONTAL, 
                                        variable=self.vpu_level,
                                        command=self.on_vpu_level_change)
        self.vpu_level_scale.pack(fill=tk.X)
        
        # ---> IMAGE PREVIEW UI <---
        preview_frame = tk.LabelFrame(tab, text="Live Image Buffer Preview", padx=10, pady=10)
        preview_frame.grid(row=2, column=0, sticky="nsew", padx=5, pady=5)
        preview_frame.columnconfigure(0, weight=1)
        
        tk.Button(preview_frame, text="📸 Fetch DDR Snapshot", command=lambda: self.send_serial("P\n"),
                  bg="lightblue", font=("Arial", 10, "bold")).grid(row=0, column=0, pady=(0, 5), sticky="ew")

        self.img_preview_lbl = tk.Label(preview_frame, text="No Image Selected\n(Browse for an image in the AI Control tab)", bg="black", fg="white", height=10)
        self.img_preview_lbl.grid(row=1, column=0, sticky="nsew")
        preview_frame.rowconfigure(1, weight=1)


    def setup_console_tab(self, notebook):
        tab = tk.Frame(notebook)
        notebook.add(tab, text="Console")
        
        self.console = scrolledtext.ScrolledText(tab, font=("Consolas", 9), height=35)
        self.console.pack(fill=tk.BOTH, expand=False, padx=15, pady=5)
        
        dbg_frame = tk.LabelFrame(tab, text="Quick Debug", padx=5, pady=5)
        dbg_frame.pack(fill=tk.X, padx=15, pady=5)
        
        cmds = [
            ("Ping", "\n"), ("Status", "S\n"), ("Soft Reset", "\x03\n"), 
            ("Test Pattern", "M200,400,200,400;\n"), ("Read Regs", "R2100000000000040;\n"), 
            ("Read BBox X", "R4002000400000004;\n"), ("List BBoxes", "L\n"), ("Clear BBox","W4003000000000000; \nC;\n")
        ]
        
        for i, (text, cmd) in enumerate(cmds):
            tk.Button(dbg_frame, text=text, command=lambda c=cmd: self.send_serial(c), width=12).grid(row=i//4, column=i%4, padx=5, pady=2)

        entry_frame = tk.Frame(tab)
        entry_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.cmd_entry = tk.Entry(entry_frame)
        self.cmd_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.cmd_entry.bind('<Return>', self.send_console_cmd)
        
        tk.Button(entry_frame, text="Send", command=self.send_console_cmd).pack(side=tk.LEFT)
        tk.Button(entry_frame, text="Clear Log", command=self.clear_console, bg="lightgrey").pack(side=tk.RIGHT, padx=5)

    def on_vpu_mode_change(self):
        mode = self.vpu_mode.get()
        if hasattr(self, 'level_frame'):
            if mode == 0:
                self.level_frame.config(text="Level (Disabled)")
                self.vpu_level_scale.config(state="disabled")
            else:
                self.level_frame.config(text="Effect Level (0-100%)")
                self.vpu_level_scale.config(state="normal")

        if mode == 0:
            self.send_serial("W21000018000007FC;\n")
        else:
            cmd = "W2100001800000FFC;\n" if mode == 1 else "W21000018000013FC;\n"
            self.send_serial(cmd)
            self.on_vpu_level_change(self.vpu_level.get())

    def on_vpu_level_change(self, val):
        val = int(float(val))
        hex_val = format(int(val * 2.55), '08X')
        self.send_serial(f"W21000038{hex_val};\n", False)

    def trigger_vpu_soft_reset(self):
        if not self.is_connected:
            self.log("System not connected. Cannot issue reset.")
            return

        def reset_task():
            self.log("Issuing VPU Soft Reset...")
            self.send_serial("W2100000000000001;\n")
            
            time.sleep(0.5) # Wait 500ms before releasing reset
            
            self.send_serial("W2100000000000000;\n")
            self.log("VPU Reset released.")

        # Run in a separate thread to prevent GUI freezing during sleep
        threading.Thread(target=reset_task, daemon=True).start()

    def send_custom_bbox(self):
        try:
            l = int(self.entry_l.get())
            r = int(self.entry_r.get())
            t = int(self.entry_t.get())
            b = int(self.entry_b.get())
            self.send_manual_cmd(l, r, t, b)
        except ValueError:
            messagebox.showerror("Error", "Please enter integer values")

    def send_manual_cmd(self, l, r, t, b):
        cmd = f"M{l},{r},{t},{b};\n"
        self.send_serial(cmd)
        self.log(f"Manual Box: L:{l} R:{r} T:{t} B:{b}")
        self.add_history("Manual", l, r, t, b)

    def add_history(self, source, l, r, t, b):
        ts = datetime.now().strftime("%H:%M:%S")
        self.bbox_tree.insert("", 0, values=(ts, source, l, r, t, b))
        if len(self.bbox_tree.get_children()) > 50:
            self.bbox_tree.delete(self.bbox_tree.get_children()[-1])

    def clear_bbox_history(self):
        for item in self.bbox_tree.get_children():
            self.bbox_tree.delete(item)
        self.send_serial("X;\n") 
        self.log("BBox History Reset (GUI & Device)")

    def update_ports_list(self):
        self.ports = [p.device for p in serial.tools.list_ports.comports()]
        
    def refresh_ports(self):
        self.update_ports_list()
        if hasattr(self, 'port_var'): 
            if self.ports: self.port_var.set(self.ports[0])

    def toggle_connection(self):
        if self.is_connected:
            self.serial_port.close()
            self.is_connected = False
            self.connect_btn.config(text="Connect", bg="lightgreen")
        else:
            try:
                self.serial_port = serial.Serial(self.port_var.get(), int(self.baud_var.get()), timeout=1)
                self.is_connected = True
                self.connect_btn.config(text="Disconnect", bg="lightcoral")
                threading.Thread(target=self.read_serial, daemon=True).start()
            except Exception as e:
                messagebox.showerror("Error", str(e))

    def send_serial(self, data, show=True):
        if self.is_connected:
            self.serial_port.write(data.encode())
            if show: self.log(f"TX: {data.strip()}")

    def send_console_cmd(self, event=None):
        cmd = self.cmd_entry.get()
        self.send_serial(cmd + "\n")
        self.cmd_entry.delete(0, tk.END)

    def read_serial(self):
        raw_buffer = bytearray()
        expected_bytes = 416 * 256 * 4 * 2

        while self.is_connected:
            try:
                if self.serial_port.in_waiting:
                    chunk = self.serial_port.read(self.serial_port.in_waiting)
                    raw_buffer.extend(chunk)

                    # --- BINARY IMAGE PROCESSING ---
                    if b"[IMG_START]" in raw_buffer:
                        _, raw_buffer = raw_buffer.split(b"[IMG_START]", 1)
                        
                        self.window.after(0, lambda: self.log("Receiving snapshot... stay still."))
                        
                        start_time = time.time()
                        # Wait until we have the full image
                        while len(raw_buffer) < expected_bytes:
                            if self.serial_port.in_waiting:
                                raw_buffer.extend(self.serial_port.read(self.serial_port.in_waiting))
                            
                            # Timeout to prevent infinite freeze
                            if time.time() - start_time > 60:
                                self.window.after(0, lambda: self.log("Error: Image Fetch Timeout"))
                                break
                            time.sleep(0.01) 
                        
                        if len(raw_buffer) >= expected_bytes:
                            # Extract exactly the image data
                            img_data = bytes(raw_buffer[:expected_bytes])
                            raw_buffer = raw_buffer[expected_bytes:]
                            
                            # Handle the trailing tag if it's there
                            if b"[IMG_END]" in raw_buffer:
                                _, raw_buffer = raw_buffer.split(b"[IMG_END]", 1)

                            self.window.after(0, self.update_live_preview, img_data)
                            self.window.after(0, lambda: self.log("Snapshot complete."))

                    # --- NORMAL TEXT PROCESSING ---
                    while b'\n' in raw_buffer:
                        line_bytes, raw_buffer = raw_buffer.split(b'\n', 1)
                        line_str = line_bytes.decode(errors='ignore').strip()
                        if line_str:
                            self.window.after(0, lambda l=line_str: self.console.insert(tk.END, l + '\n'))
                            self.window.after(0, lambda: self.console.see(tk.END))
                            self.window.after(0, lambda l=line_str: self.parse_status(l))

            except Exception as e:
                print(f"Serial Read Error: {e}")
                self.is_connected = False
                break

    def update_live_preview(self, raw_data):
        try:
            # 1. New Dimensions from the script
            width, height = 416, 256
            
            # 2. Interpret as Float16 (2 bytes per value)
            # This matches the 'pixels.tobytes()' in your script
            pixel_array = np.frombuffer(raw_data, dtype=np.float16)
            
            # 3. Reshape to (Pixels, Channels)
            # Script uses R, G, B, PAD
            pixel_matrix = pixel_array.reshape((height, width, 4))
            
            # 4. Extract only R, G, B (Drop the 4th PAD channel)
            rgb_matrix = pixel_matrix[:, :, :3]
            
            # 5. Convert Float16 (0.0 - 1.0) back to Uint8 (0 - 255) for Display
            # We clip to ensure values stay in range
            rgb_matrix = np.clip(rgb_matrix * 255.0, 0, 255).astype(np.uint8)
            
            # 6. Render
            pil_img = Image.fromarray(rgb_matrix, 'RGB')
            
            # Resize for the UI (Optional)
            pil_img = pil_img.resize((640, 360), Image.NEAREST) 
            
            self.preview_photo = ImageTk.PhotoImage(pil_img) 
            self.img_preview_lbl.config(image=self.preview_photo, text="")
            
        except Exception as e:
            self.log(f"Live preview rendering error: {e}")

    def parse_status(self, text):
        if "[SD_LOAD_DONE]" in text:
            self.window.after(0, self.sd_load_complete)
            
        if "weight_file.bin:" in text:
            self.status_vars["weight"].set("Weight: ✓" if "UPLOADED" in text else "Weight: ✗")
        if "padding_file.bin:" in text:
            self.status_vars["padding"].set("Padding: ✓" if "UPLOADED" in text else "Padding: ✗")
        if "Current Image:" in text and "✓" in text:
            self.status_vars["image"].set("Image: ✓")
        
        if "Boundaries: L:" in text:
            try:
                parts = text.split("Boundaries: L:")[1].split()
                l = parts[0]
                r = parts[1].split(":")[1]
                t = parts[2].split(":")[1]
                b = parts[3].split(":")[1]
                self.window.after(0, lambda: self.add_history("AI", l, r, t, b))
            except: pass
            
        if "DATA:" in text:
            try:
                val = text.split("DATA:")[1].strip().split()[0]
                if hasattr(self, 'last_mem_check_index') and self.last_mem_check_index is not None:
                    self.mem_monitors[self.last_mem_check_index]["var"].set(val)
                    self.last_mem_check_index = None 
            except: pass

    def log(self, msg):
        ts = datetime.now().strftime("%H:%M:%S")
        full_msg = f"[{ts}] {msg}"
        self.console.insert(tk.END, full_msg + "\n")
        self.console.see(tk.END)
        self.var_latest_event.set(full_msg)

    def clear_console(self):
        self.console.delete(1.0, tk.END)

if __name__ == "__main__":
    app = AIBoundingBoxGUI()
    app.window.mainloop()
