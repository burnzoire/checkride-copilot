-- Checkride Copilot — append this block to the END of:
--   %USERPROFILE%\Saved Games\DCS\Scripts\export.lua
--
-- Uses safe chaining — existing exports (Tacview, SRS, SimShaker, etc.) are unaffected.
-- Sends JSON to localhost:7778 (UDP) at ~4 Hz.
--
-- Confirmed arg numbers sourced from DCS-BIOS/lib/FA-18C_hornet.lua.
-- Unknowns (radar sub-mode, TGP tracking, selected weapon, laser) are included
-- as raw CockpitParams for Phase 0 discovery in the monitor tool.

do
    local _cc = {}

    _cc.socket   = require("socket")
    _cc.JSON     = loadfile("Scripts/JSON.lua")()
    _cc.host     = "127.0.0.1"
    _cc.port     = 7778
    _cc.udp      = _cc.socket.udp()
    _cc.interval = 0.25   -- 4 Hz
    _cc.last_ts  = 0

    _cc.udp:settimeout(0)

    -- Confirmed arg indices (from DCS-BIOS FA-18C_hornet.lua)
    local ARG = {
        MASTER_ARM      = 49,   -- 0.0=SAFE, 1.0=ARM
        RADAR_SW        = 440,  -- 0.0=OFF, 0.1=STBY, 0.2=OPR, 0.3=EMERG
        FLIR_SW         = 439,  -- TGP/pod power: 0.0=ON, 0.5=STBY, 1.0=OFF
        BATTERY_SW      = 404,  -- 0.0=ON, 0.5=OFF, 1.0=ORIDE
        APU_CONTROL     = 375,  -- 0.0=OFF, 1.0=ON
        GEAR_LEVER      = 226,  -- 0.0=UP, 1.0=DOWN
        FLAP_SW         = 234,  -- 0.0=AUTO, 0.5=HALF, 1.0=FULL
        HOOK_LEVER      = 293,  -- 0.0=UP, 1.0=DOWN
        ENGINE_CRANK    = 377,  -- rocker: LEFT/OFF/RIGHT
        MASTER_MODE_AA  = 458,  -- pushbutton state (0/1)
        MASTER_MODE_AG  = 459,  -- pushbutton state (0/1)
        HMD_BRT         = 136,  -- 0–0.75; >0 = HMD powered

        -- UNCONFIRMED — indices to discover via monitor --diff
        -- RADAR_MODE_SUB  = ???  RWS/TWS/STT displayed on radar DDI
        -- TGP_TDC_PRI     = ???  which DDI has TDC priority
        -- TGP_TRACKING    = ???  TGP has tracked designation
        -- LASER_ARMED     = ???  laser trigger armed
        -- SELECTED_WEAPON = ???  SMS selected store
    }

    local function _cc_arg(dev0, idx)
        local ok, v = pcall(function() return dev0:get_argument_value(idx) end)
        return ok and v or nil
    end

    local function _cc_radar_power(v)
        if v == nil then return "unknown" end
        if v < 0.05 then return "OFF" end
        if v < 0.15 then return "STBY" end
        if v < 0.25 then return "OPR" end
        return "EMERG"
    end

    local function _cc_flir_power(v)
        if v == nil then return "unknown" end
        if v < 0.25 then return "ON" end
        if v < 0.75 then return "STBY" end
        return "OFF"
    end

    local function _cc_battery(v)
        if v == nil then return "unknown" end
        if v < 0.25 then return "ON" end
        if v < 0.75 then return "OFF" end
        return "ORIDE"
    end

    local function _cc_gear(v)
        if v == nil then return "unknown" end
        return v > 0.5 and "DOWN" or "UP"
    end

    local function _cc_flaps(v)
        if v == nil then return "unknown" end
        if v < 0.25 then return "AUTO" end
        if v < 0.75 then return "HALF" end
        return "FULL"
    end

    local function _cc_master_arm(v)
        if v == nil then return "unknown" end
        return v > 0.5 and "ARM" or "SAFE"
    end

    local function _cc_get_cockpit(dev0)
        local cp = {}

        -- Confirmed reads
        local ma   = _cc_arg(dev0, ARG.MASTER_ARM)
        local rad  = _cc_arg(dev0, ARG.RADAR_SW)
        local flir = _cc_arg(dev0, ARG.FLIR_SW)
        local bat  = _cc_arg(dev0, ARG.BATTERY_SW)
        local apu  = _cc_arg(dev0, ARG.APU_CONTROL)
        local gear = _cc_arg(dev0, ARG.GEAR_LEVER)
        local flap = _cc_arg(dev0, ARG.FLAP_SW)
        local hook = _cc_arg(dev0, ARG.HOOK_LEVER)
        local hmd  = _cc_arg(dev0, ARG.HMD_BRT)
        local maa  = _cc_arg(dev0, ARG.MASTER_MODE_AA)
        local mag  = _cc_arg(dev0, ARG.MASTER_MODE_AG)

        cp.master_arm      = _cc_master_arm(ma)
        cp.radar_power_sw  = _cc_radar_power(rad)
        cp.tgp_power_sw    = _cc_flir_power(flir)
        cp.battery_switch  = _cc_battery(bat)
        cp.apu_running     = apu ~= nil and apu > 0.5 or false
        cp.gear_lever      = _cc_gear(gear)
        cp.flaps_sw        = _cc_flaps(flap)
        cp.hook_lever      = _cc_gear(hook)  -- same binary logic
        cp.hmd_enabled     = hmd ~= nil and hmd > 0.05 or false
        cp.master_mode     = (maa ~= nil and maa > 0.5) and "AA" or
                             (mag ~= nil and mag > 0.5) and "AG" or "NAV"

        -- Raw dump of ALL args for Phase 0 discovery.
        -- The monitor --diff tool will show which indices change as you toggle
        -- radar sub-mode, TGP TDC priority, laser arm, weapon select, etc.
        -- Once confirmed, those indices get promoted to named reads above.
        local ok_cp, raw = pcall(LoGetCockpitParams)
        if ok_cp and raw then
            cp._raw = raw
        end

        return cp
    end

    -- Chain into any existing LuaExportAfterNextFrame
    local _cc_prev = LuaExportAfterNextFrame

    LuaExportAfterNextFrame = function()
        if _cc_prev then pcall(_cc_prev) end

        local now = LoGetModelTime()
        if (now - _cc.last_ts) < _cc.interval then return end
        _cc.last_ts = now

        local ok_self,    self_data    = pcall(LoGetSelfData)
        local ok_mech,    mech_info    = pcall(LoGetMechInfo)
        local ok_nav,     nav_info     = pcall(LoGetNavigationInfo)
        local ok_payload, payload_info = pcall(LoGetPayloadInfo)
        local ok_eng,     eng_info     = pcall(LoGetEngineInfo)
        local ok_ias,     ias_ms       = pcall(LoGetIndicatedAirSpeed)
        local ok_tas,     tas_ms       = pcall(LoGetTrueAirSpeed)
        local ok_mach,    mach_val     = pcall(LoGetMachNumber)
        local ok_aoa,     aoa_val      = pcall(LoGetAngleOfAttack)
        local ok_fuel,    fuel_kg      = pcall(LoGetFuelMass)

        local dev0 = GetDevice(0)
        local cockpit = (dev0 ~= nil) and _cc_get_cockpit(dev0) or {}

        local packet = {
            event = "export",
            ts    = now,
            data  = {
                SelfData      = ok_self    and self_data    or {},
                MechInfo      = ok_mech    and mech_info    or {},
                NavInfo       = ok_nav     and nav_info     or {},
                PayloadInfo   = ok_payload and payload_info or {},
                EngineInfo    = ok_eng     and eng_info     or nil,
                FlightData    = {
                    IAS_ms    = ok_ias  and ias_ms   or nil,
                    TAS_ms    = ok_tas  and tas_ms   or nil,
                    Mach      = ok_mach and mach_val or nil,
                    AoA_deg   = ok_aoa  and aoa_val  or nil,
                    FuelKg    = ok_fuel and fuel_kg  or nil,
                },
                CockpitParams = cockpit,
            }
        }

        local ok_json, json_str = pcall(_cc.JSON.encode, _cc.JSON, packet)
        if ok_json and json_str then
            _cc.udp:sendto(json_str, _cc.host, _cc.port)
        end
    end

    local _cc_prev_start = LuaExportStart
    LuaExportStart = function()
        if _cc_prev_start then pcall(_cc_prev_start) end
    end

    local _cc_prev_stop = LuaExportStop
    LuaExportStop = function()
        if _cc_prev_stop then pcall(_cc_prev_stop) end
        pcall(function() _cc.udp:close() end)
    end
end
