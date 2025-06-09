#!/usr/bin/env python3

import re
import math
import webbrowser
from textual.app import App, ComposeResult
from textual.widgets import Input, Static, Header
from textual.containers import Vertical, Horizontal
from textual import events

try:
    from pint import UnitRegistry
    ureg = UnitRegistry()
    PINT_AVAILABLE = True
except ImportError:
    PINT_AVAILABLE = False

class MultiTool(App):
    """Multi-purpose calculator, unit converter, and search tool with dynamic tiles."""

    def __init__(self):
        super().__init__()
        self.theme = "catppuccin-mocha"
        self.close_after_search = False
            # True  = close app after searching
            # False = keep app open after searching
        
        # Custom search patterns - add your own here!
        self.search_patterns = {
            r'jira ([A-Z]+-\d+)': 'https://company.atlassian.net/browse/{0}',
            r'wiki (.+)': 'https://internal.wiki.com/search?q={0}',
            r'gh (.+)': 'https://github.com/search?q={0}',
            r'docs (.+)': 'https://docs.company.com/search?q={0}',
            r'stack (.+)': 'https://stackoverflow.com/search?q={0}',
            r'py (.+)': 'https://docs.python.org/3/search.html?q={0}',
        }
        
        self.unit_aliases = {
            'meter': 'm', 'meters': 'm', 'metre': 'm', 'metres': 'm',
            'centimeter': 'cm', 'centimeters': 'cm', 'centimetre': 'cm', 'centimetres': 'cm',
            'millimeter': 'mm', 'millimeters': 'mm', 'millimetre': 'mm', 'millimetres': 'mm',
            'kilometer': 'km', 'kilometers': 'km', 'kilometre': 'km', 'kilometres': 'km',
            'inch': 'in', 'inches': 'in',
            'foot': 'ft', 'feet': 'ft',
            'yard': 'yd', 'yards': 'yd',
            'mile': 'mi', 'miles': 'mi',
            'gram': 'g', 'grams': 'g',
            'kilogram': 'kg', 'kilograms': 'kg', 'kilo': 'kg', 'kilos': 'kg',
            'pound': 'lb', 'pounds': 'lb', 'lbs': 'lb',
            'ounce': 'oz', 'ounces': 'oz',
            'celsius': 'c', 'centigrade': 'c',
            'fahrenheit': 'f',
            'kelvin': 'k',
            'rankine': 'r',
            'g/cc': 'g/cm3',
            'kg/cc': 'kg/cm3',
            'lb/cc': 'lb/cm3',
            'oz/cc': 'oz/cm3',
            'watt': 'w', 'watts': 'w',
            'horsepower': 'hp', 'horse-power': 'hp',
            'kilowatt': 'kw', 'kilowatts': 'kw', 'kilo-watt': 'kw', 'kilo-watts': 'kw',
            'newton-meter': 'n-m', 'newton-metre': 'n-m', 'newtonmeter': 'n-m', 'newtonmetre': 'n-m',
            'foot-pound': 'ft-lb', 'footpound': 'ft-lb', 'foot-pounds': 'ft-lb', 'footpounds': 'ft-lb',
            'inch-pound': 'in-lb', 'inchpound': 'in-lb', 'inch-pounds': 'in-lb', 'inchpounds': 'in-lb',
            'milliliter': 'ml', 'milliliters': 'ml', 'millilitre': 'ml', 'millilitres': 'ml',
            'cubic-inch': 'in^3', 'cubic-inches': 'in^3', 'cubicinch': 'in^3', 'cubicinches': 'in^3',
            'cubic-centimeter': 'cm^3', 'cubic-centimeters': 'cm^3', 'cubicentimeter': 'cm^3',
            'tablespoon': 'tbsp', 'tablespoons': 'tbsp', 'table-spoon': 'tbsp', 'table-spoons': 'tbsp',
            'teaspoon': 'tsp', 'teaspoons': 'tsp', 'tea-spoon': 'tsp', 'tea-spoons': 'tsp',
            'quart': 'qt', 'quarts': 'qt',
            'pint': 'pt', 'pints': 'pt'
        }

        self.length_meters = {
            'mm': 0.001, 'cm': 0.01, 'm': 1.0, 'km': 1000,
            'in': 0.0254, 'ft': 0.3048, 'yd': 0.9144, 'mi': 1609.344
        }

        self.mass_kg = {
            'g': 0.001, 'kg': 1.0, 'lb': 0.453592, 'oz': 0.0283495
        }
        
        self.density_kg_per_m3 = {
            'g/mm3': 1_000_000.0, 'g/cm3': 1_000.0, 'g/m3': 0.001, 'g/l': 1.0, 'g/ft3': 0.0353147, 'g/in3': 61.0237,
            'kg/mm3': 1_000_000_000.0, 'kg/cm3': 1_000_000.0, 'kg/m3': 1.0, 'kg/l': 1000.0, 'kg/gal': 264.172, 'kg/ft3': 35.3147, 'kg/in3': 61023.7,
            'lb/mm3': 453_592_000.0, 'lb/cm3': 453_592.0, 'lb/m3': 0.453592, 'lb/l': 453.592, 'lb/gal': 119.826, 'lb/ft3': 16.0185, 'lb/in3': 27679.9,
            'oz/in3': 1729.99, 'oz/ft3': 1.00115, 'oz/gal': 7.48915,
        }

    CSS = """
    Screen { align: center middle; background: black; }
    #container { width: 60; max-width: 80; height: auto; background: $surface; border: solid $primary; }
    #input { height: 3; margin: 1; width: 100%; }
    #tiles-container { height: 8; width: 100%; }
    .tile { display: none; height: auto; margin: 0 1 1 1; background: $panel; color: $text; padding: 1; width: 100%; border: solid $primary; }
    .tile.visible { display: block; }
    #instructions { dock: bottom; height: 1; background: $surface; color: $text-muted; padding: 0 1; width: 100%; }
    #instructions-left { text-align: left; width: 50%; }
    #instructions-right { text-align: right; width: 50%; }
    """
    
    def compose(self) -> ComposeResult:
        with Vertical(id="container"):
            yield Header(show_clock=False)
            yield Input(placeholder="Type a conversion, calculation, or search...", id="input")
            with Vertical(id="tiles-container"):
                yield Static("", id="unit-tile", classes="tile")
                yield Static("", id="calc-tile", classes="tile")
                yield Static("", id="search-tile", classes="tile")
            with Horizontal(id="instructions"):
                yield Static("ESC: Exit", id="instructions-left")
                yield Static("Ctrl+N: Clear", id="instructions-right")
    
    def on_mount(self) -> None:
        input_widget = self.query_one("#input", Input)
        input_widget.focus()
        input_widget.action_submit = lambda: None
    
    def show_tile(self, tile_id: str, content: str) -> None:
        tile = self.query_one(f"#{tile_id}")
        tile.update(content)
        tile.add_class("visible")
    
    def hide_tile(self, tile_id: str) -> None:
        tile = self.query_one(f"#{tile_id}")
        tile.remove_class("visible")
    
    def hide_all_tiles(self) -> None:
        for tile_id in ["unit-tile", "calc-tile", "search-tile"]:
            self.hide_tile(tile_id)
    
    def check_custom_search(self, text: str) -> tuple[str, str] | None:
        """Check if text matches any custom search patterns.
        Returns (pattern_name, url) if match found, None otherwise."""
        for pattern, url_template in self.search_patterns.items():
            match = re.match(pattern, text.strip(), re.IGNORECASE)
            if match:
                # Extract the captured group and format the URL
                search_term = match.group(1)
                formatted_url = url_template.format(search_term.replace(' ', '+'))
                # Extract a friendly name from the pattern for display
                pattern_name = pattern.split('(')[0].strip('r\'').replace('\\', '')
                return (pattern_name, formatted_url)
        return None
    
    def preprocess_volume_notation(self, text: str) -> str:
        """Convert volume notations like 5m3, 5 m3, 5ft3 to proper cubic units."""
        # Handle patterns like 5m3, 5 m3, 5ft3, 5 ft3, etc.
        volume_pattern = r'(\d+(?:\.\d+)?)\s*([a-zA-Z]+)3\b'
        
        def replace_volume(match):
            number = match.group(1)
            unit = match.group(2)
            return f"{number} {unit}^3"
        
        return re.sub(volume_pattern, replace_volume, text)
    
    def on_input_changed(self, event: Input.Changed) -> None:
        text = event.value.strip()
        self.hide_all_tiles()
        if not text:
            return
        
        # Check for custom search patterns first
        custom_search = self.check_custom_search(text)
        if custom_search:
            pattern_name, url = custom_search
            self.show_tile("search-tile", f"Press Ctrl+Enter\nto search {pattern_name}:\n{url}")
            return
        
        unit_result = self.try_unit_conversion(text)
        if unit_result:
            self.show_tile("unit-tile", unit_result)
            return
        
        calc_result = self.try_math_calculation(text)
        if calc_result:
            self.show_tile("calc-tile", calc_result)
            return
        
        self.show_tile("search-tile", "No matches here\nPress `Ctrl+Enter` to search Google")
    
    def try_unit_conversion(self, text: str) -> str | None:
        # Preprocess volume notation before trying Pint
        preprocessed_text = self.preprocess_volume_notation(text)
        
        # Try Pint first if available - it handles way more units
        if PINT_AVAILABLE:
            pint_result = self.try_pint_conversion(preprocessed_text)
            if pint_result:
                return pint_result
        
        # Fall back to built-in conversions
        explicit_result = self.convert_units_explicit(preprocessed_text)
        if explicit_result:
            return explicit_result
        
        auto_result = self.convert_units_auto(preprocessed_text)
        if auto_result:
            return auto_result
        
        return None
    
    def try_pint_conversion(self, text: str) -> str | None:
        """Try to convert using Pint library for comprehensive unit support."""
        try:
            # Handle explicit conversions like "5 feet to meters"
            to_patterns = [
                r'(.+?)\s+(?:to|in|as)\s+(.+)',
                r'(.+?)\s*->\s*(.+)',
            ]
            
            for pattern in to_patterns:
                match = re.match(pattern, text.strip(), re.IGNORECASE)
                if match:
                    from_str, to_str = match.groups()
                    try:
                        quantity = ureg(from_str.strip())
                        result = quantity.to(to_str.strip())
                        return f"{quantity:~P} = {result:~P.6g}"
                    except:
                        continue
            
            # Handle auto-conversions for single quantities
            try:
                quantity = ureg(text.strip())
                # Get the base unit type and suggest common conversions
                dimensionality = quantity.dimensionality
                
                conversions = []
                
                # Length conversions
                if dimensionality == ureg.meter.dimensionality:
                    targets = ['meter', 'foot', 'inch', 'centimeter']
                    for target in targets:
                        try:
                            if quantity.units != ureg(target).units:
                                result = quantity.to(target)
                                conversions.append(f"{result:~P.6g}")
                        except:
                            continue
                
                # Mass conversions
                elif dimensionality == ureg.kilogram.dimensionality:
                    targets = ['kilogram', 'pound', 'gram', 'ounce']
                    for target in targets:
                        try:
                            if quantity.units != ureg(target).units:
                                result = quantity.to(target)
                                conversions.append(f"{result:~P.6g}")
                        except:
                            continue
                
                # Temperature conversions
                elif dimensionality == ureg.kelvin.dimensionality:
                    targets = ['celsius', 'fahrenheit', 'kelvin', 'rankine']
                    for target in targets:
                        try:
                            if quantity.units != ureg(target).units:
                                result = quantity.to(target)
                                conversions.append(f"{result:~P.6g}")
                        except:
                            continue
                
                # Energy conversions
                elif dimensionality == ureg.joule.dimensionality:
                    targets = ['joule', 'calorie', 'BTU', 'kilowatt_hour']
                    for target in targets:
                        try:
                            if quantity.units != ureg(target).units:
                                result = quantity.to(target)
                                conversions.append(f"{result:~P.6g}")
                        except:
                            continue
                
                # Pressure conversions
                elif dimensionality == ureg.pascal.dimensionality:
                    targets = ['pascal', 'bar', 'psi', 'atmosphere']
                    for target in targets:
                        try:
                            if quantity.units != ureg(target).units:
                                result = quantity.to(target)
                                conversions.append(f"{result:~P.6g}")
                        except:
                            continue
                
                # Volume conversions
                elif '[length] ** 3' in str(dimensionality):
                    targets = ['liter', 'gallon', 'cubic_meter', 'fluid_ounce', 'cubic_foot']
                    for target in targets:
                        try:
                            if quantity.units != ureg(target).units:
                                result = quantity.to(target)
                                conversions.append(f"{result:~P.6g}")
                        except:
                            continue
                
                # Power conversions
                elif dimensionality == ureg.watt.dimensionality:
                    targets = ['watt', 'kilowatt', 'horsepower', 'BTU_per_hour']
                    for target in targets:
                        try:
                            if quantity.units != ureg(target).units:
                                result = quantity.to(target)
                                conversions.append(f"{result:~P.6g}")
                        except:
                            continue

                # Torque conversions  
                elif dimensionality == ureg('newton * meter').dimensionality:
                    targets = ['newton_meter', 'foot_pound_force', 'inch_pound_force']
                    for target in targets:
                        try:
                            if quantity.units != ureg(target).units:
                                result = quantity.to(target)
                                conversions.append(f"{result:~P.6g}")
                        except:
                            continue
                
                return "\n".join(conversions[:3]) if conversions else None  # Limit to 3 conversions
                
            except:
                return None
                
        except Exception:
            return None
        
        return None
    
    def try_math_calculation(self, text: str) -> str | None:
        try:
            calc_text = text.lower()
            # Fixed math function replacements - need to use math.asin, etc.
            replacements = {
                "sqrt": "math.sqrt", 
                "asin": "math.asin", 
                "acos": "math.acos", 
                "atan": "math.atan", 
                "sin": "math.sin", 
                "cos": "math.cos", 
                "tan": "math.tan", 
                "log": "math.log", 
                "ln": "math.log",  # Natural log alias
                "log10": "math.log10",
                "pi": "math.pi",
                "e": "math.e"
            }
            
            # Check if expression contains trig functions
            trig_functions = ["sin", "cos", "tan", "asin", "acos", "atan"]
            has_trig = any(trig_func in text.lower() for trig_func in trig_functions)
            
            # Replace functions but be careful about order (longer names first)
            sorted_replacements = sorted(replacements.items(), key=lambda x: len(x[0]), reverse=True)
            for old, new in sorted_replacements:
                # Use word boundaries to avoid partial replacements
                calc_text = re.sub(rf'\b{re.escape(old)}\b', new, calc_text)

            # Handle exponents (^ to **)
            calc_text = calc_text.replace('^', '**')

            if any(char in calc_text for char in "+-*/()") or any(func in calc_text for func in replacements.values()) or '**' in calc_text:
                # Auto-close parentheses
                open_parens = calc_text.count('(')
                close_parens = calc_text.count(')')
                if open_parens > close_parens:
                    calc_text += ')' * (open_parens - close_parens)
                
                if has_trig:
                    # Calculate both degrees and radians versions
                    try:
                        # For degrees: convert input to radians for trig functions, convert output from radians for inverse trig
                        deg_calc_text = calc_text
                        # Convert degree inputs to radians for forward trig functions
                        deg_calc_text = re.sub(r'math\.(sin|cos|tan)\(([^)]+)\)', r'math.\1(math.radians(\2))', deg_calc_text)
                        # Convert radian outputs to degrees for inverse trig functions
                        deg_calc_text = re.sub(r'math\.(asin|acos|atan)\(([^)]+)\)', r'math.degrees(math.\1(\2))', deg_calc_text)
                        
                        deg_result = eval(deg_calc_text, {"__builtins__": {}, "math": math})
                        rad_result = eval(calc_text, {"__builtins__": {}, "math": math})
                        
                        deg_formatted = f"{deg_result:.10g}" if isinstance(deg_result, float) else f"{deg_result}"
                        rad_formatted = f"{rad_result:.10g}" if isinstance(rad_result, float) else f"{rad_result}"
                        
                        return f"= {deg_formatted} (degrees)\n= {rad_formatted} (radians)"
                    except:
                        # If degree conversion fails, fall back to radians only
                        result = eval(calc_text, {"__builtins__": {}, "math": math})
                        return f"= {result:.10g}" if isinstance(result, float) else f"= {result}"
                else:
                    # No trig functions, calculate normally
                    result = eval(calc_text, {"__builtins__": {}, "math": math})
                    return f"= {result:.10g}" if isinstance(result, float) else f"= {result}"
        except Exception:
            return None
        return None
    
    def convert_units_explicit(self, text: str) -> str | None:
        patterns = [
            r'(\d+(?:\.\d+)?)\s*([a-zA-Z\/³\d\^]+)\s+(?:to|in|as)\s+([a-zA-Z\/³\d\^]+)',
            r'(\d+(?:\.\d+)?)\s*([a-zA-Z\/³\d\^]+)\s*->\s*([a-zA-Z\/³\d\^]+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                value, from_unit, to_unit = match.groups()
                return self.do_conversion(float(value), from_unit.lower(), to_unit.lower())
        return None
    
    def convert_units_auto(self, text: str) -> str | None:
        pattern = r'^(\d+(?:\.\d+)?)\s*([a-zA-Z\/³\d\^]+)$'
        match = re.match(pattern, text.strip(), re.IGNORECASE)
        if not match:
            return None

        value, unit = match.groups()
        value = float(value)
        original_unit = unit.lower()
        unit = self.unit_aliases.get(original_unit, original_unit)

        target_units = []
        if unit in self.density_kg_per_m3:
            target_units = ['kg/m3', 'g/cm3', 'lb/ft3', 'lb/in3']
        elif unit in self.length_meters:
            target_units = ['m', 'cm', 'ft', 'in']
        elif unit in self.mass_kg:
            target_units = ['kg', 'g', 'lb', 'oz']
        elif unit in ['c', 'f', 'k', 'r']:
            if unit == 'c':
                target_units = ['f', 'k', 'r']
            elif unit == 'f':
                target_units = ['c', 'k', 'r']
            elif unit == 'k':
                target_units = ['c', 'f', 'r']
            elif unit == 'r':
                target_units = ['c', 'f', 'k']

        if not target_units:
            return None

        conversions = []
        for target_unit in target_units:
            if unit == target_unit:
                continue

            result = self.do_conversion(value, unit, target_unit)
            if result and "Don't know" not in result:
                parts = result.split(' = ')
                if len(parts) == 2:
                    conversions.append(parts[1])

        return "\n".join(conversions) if conversions else None

    def do_conversion(self, value: float, from_unit: str, to_unit: str) -> str:
        from_unit = self.unit_aliases.get(from_unit, from_unit)
        to_unit = self.unit_aliases.get(to_unit, to_unit)

        # Enhanced temperature conversions with Kelvin and Rankine
        if from_unit in ['c', 'f', 'k', 'r'] and to_unit in ['c', 'f', 'k', 'r']:
            if from_unit == 'c':
                if to_unit == 'f':
                    result = value * 9/5 + 32
                    return f"{value}°C = {result:.2f}°F"
                elif to_unit == 'k':
                    result = value + 273.15
                    return f"{value}°C = {result:.2f}K"
                elif to_unit == 'r':
                    result = (value + 273.15) * 9/5
                    return f"{value}°C = {result:.2f}°R"
            elif from_unit == 'f':
                if to_unit == 'c':
                    result = (value - 32) * 5/9
                    return f"{value}°F = {result:.2f}°C"
                elif to_unit == 'k':
                    result = (value - 32) * 5/9 + 273.15
                    return f"{value}°F = {result:.2f}K"
                elif to_unit == 'r':
                    result = value + 459.67
                    return f"{value}°F = {result:.2f}°R"
            elif from_unit == 'k':
                if to_unit == 'c':
                    result = value - 273.15
                    return f"{value}K = {result:.2f}°C"
                elif to_unit == 'f':
                    result = (value - 273.15) * 9/5 + 32
                    return f"{value}K = {result:.2f}°F"
                elif to_unit == 'r':
                    result = value * 9/5
                    return f"{value}K = {result:.2f}°R"
            elif from_unit == 'r':
                if to_unit == 'c':
                    result = (value - 491.67) * 5/9
                    return f"{value}°R = {result:.2f}°C"
                elif to_unit == 'f':
                    result = value - 459.67
                    return f"{value}°R = {result:.2f}°F"
                elif to_unit == 'k':
                    result = value * 5/9
                    return f"{value}°R = {result:.2f}K"
        
        if from_unit in self.length_meters and to_unit in self.length_meters:
            meters = value * self.length_meters[from_unit]
            result = meters / self.length_meters[to_unit]
            return f"{value} {from_unit} = {result:.6g} {to_unit}"
        
        if from_unit in self.mass_kg and to_unit in self.mass_kg:
            kg = value * self.mass_kg[from_unit]
            result = kg / self.mass_kg[to_unit]
            return f"{value} {from_unit} = {result:.6g} {to_unit}"

        if from_unit in self.density_kg_per_m3 and to_unit in self.density_kg_per_m3:
            base_density = value * self.density_kg_per_m3[from_unit]
            result = base_density / self.density_kg_per_m3[to_unit]
            return f"{value} {from_unit} = {result:.6g} {to_unit}"
        
        return f"Don't know how to convert {from_unit} to {to_unit}"
    
    def on_input_submitted(self, event: Input.Submitted) -> None:
        event.prevent_default()
        event.stop()
    
    def handle_word_deletion(self) -> None:
        input_widget = self.query_one("#input", Input)
        text = input_widget.value
        cursor = input_widget.cursor_position
        
        if cursor > 0:
            pos = cursor - 1
            while pos >= 0 and text[pos] in ' \t': pos -= 1
            while pos >= 0 and text[pos] not in ' \t': pos -= 1
            word_start = pos + 1
            input_widget.value = text[:word_start] + text[cursor:]
            input_widget.cursor_position = word_start
    
    def on_key(self, event: events.Key) -> None:
        if event.key == "escape":
            self.exit()
        elif event.key == "ctrl+a":
            input_widget = self.query_one("#input", Input)
            input_widget.selection = (0, len(input_widget.value))
            event.prevent_default()
        elif event.key == "ctrl+w":
            self.handle_word_deletion()
            event.prevent_default()
        elif event.key == "ctrl+n":
            self.query_one("#input", Input).value = ""
            self.hide_all_tiles()
            event.prevent_default()
        elif event.key == "ctrl+enter":
            input_text = self.query_one("#input", Input).value.strip()
            if input_text:
                # Check for custom search patterns
                custom_search = self.check_custom_search(input_text)
                if custom_search:
                    _, url = custom_search
                    webbrowser.open(url)
                else:
                    # Default to Google search
                    webbrowser.open(f"https://www.google.com/search?q={input_text.replace(' ', '+')}")
                
                if self.close_after_search:
                    self.exit()

if __name__ == "__main__":
    MultiTool().run()
